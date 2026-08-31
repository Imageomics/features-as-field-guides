"""
Latent-level scoring for Cambridge mimic-pair feature triage.

Scores every SAE latent for every mimic pair task using simple atomic metrics
(AUROC, support, mean activation per class). Vectorized across all latents via
Mann-Whitney U through rankdata.
"""

import dataclasses
import logging
import pathlib

import beartype
import numpy as np
import polars as pl
import scipy.sparse
from scipy.stats import rankdata
from tdiscovery.classification import apply_grouping, load_image_labels

from . import tasks

logger = logging.getLogger(__name__)

DEFAULT_PAIR_SPECS: list[tuple[str, str]] = [
    ("lativitta", "malleti"),
    ("cyrbia", "cythera"),
    ("notabilis", "plesseni"),
    ("hydara", "melpomene"),
    ("venus", "vulcanus"),
    ("ssp.nov.P", "malleti"),
]
DEFAULT_VIEWS = ["dorsal", "ventral"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Score all SAE latents for mimic pair discrimination."""

    run_root_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/saev/runs"
    )
    """Root directory containing SAE run folders."""
    shards_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/a6be28a1"
    )
    """Shards directory used for labels."""
    run_ids: list[str] = dataclasses.field(default_factory=list)
    """Run IDs to score. Required."""
    pair_specs: list[tuple[str, str]] = dataclasses.field(
        default_factory=lambda: DEFAULT_PAIR_SPECS.copy()
    )
    """(erato_ssp, melp_ssp) pairs."""
    views: list[str] = dataclasses.field(default_factory=lambda: DEFAULT_VIEWS.copy())
    """Views to score."""
    min_samples: int = 10
    """Minimum images per class to include a task."""
    feature_chunk: int = 1024
    """Features per AUROC chunk (controls peak memory)."""
    force_recompute: bool = False
    """If True, recompute even when parquet already exists."""

    # Slurm.
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    mem_gb: int = 80
    """Node memory in GB."""
    log_to: str = "logs"
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    include: np.ndarray
    erato_mask: np.ndarray
    melp_mask: np.ndarray
    binary: np.ndarray
    n_pos: int
    n_neg: int


@beartype.beartype
def build_task_specs(
    sv: list[str],
    *,
    pair_specs: list[tuple[str, str]],
    views: list[str],
    min_samples: int,
) -> list[TaskSpec]:
    specs = []
    for e, m in pair_specs:
        for v in views:
            task_name = tasks.get_task_name(e, m, v)
            grouping = tasks.make_label_grouping(task_name)
            targets, names, include = apply_grouping(sv, grouping)
            c2i = {name: i for i, name in enumerate(names)}
            if "erato" not in c2i or "melpomene" not in c2i:
                continue
            inc_tgt = targets[include]
            em = inc_tgt == c2i["erato"]
            mm = inc_tgt == c2i["melpomene"]
            if em.sum() < min_samples or mm.sum() < min_samples:
                continue
            specs.append(
                TaskSpec(
                    name=task_name,
                    include=include,
                    erato_mask=em,
                    melp_mask=mm,
                    binary=mm.astype(np.int8),
                    n_pos=int(mm.sum()),
                    n_neg=int(em.sum()),
                )
            )
    return specs


@beartype.beartype
def batch_wasserstein1(e_acts: np.ndarray, m_acts: np.ndarray) -> np.ndarray:
    """Wasserstein-1 distance per latent between two activation columns.

    Both inputs share the same number of latent columns but may have
    different row counts per class. Computed as the L1 area between the
    empirical CDFs over the merged, per-column sorted values.
    """
    n_e, d = e_acts.shape
    n_m = m_acts.shape[0]
    combined = np.concatenate([e_acts, m_acts], axis=0)
    n_total = n_e + n_m
    order = np.argsort(combined, axis=0)
    z = np.take_along_axis(combined, order, axis=0)
    is_e = order < n_e
    row = np.arange(1, n_total + 1)[:, None]
    cum_e = np.cumsum(is_e, axis=0)
    fe = cum_e / n_e
    fm = (row - cum_e) / n_m
    dz = np.diff(z, axis=0)
    return (np.abs(fe - fm)[:-1] * dz).sum(axis=0)


@beartype.beartype
def firing_wasserstein1(e_acts: np.ndarray, m_acts: np.ndarray) -> np.ndarray:
    """Wasserstein-1 distance per latent over firing (>0) activations only.

    NaN for latents where either class has no positive values.
    """
    out = np.empty(e_acts.shape[1], dtype=np.float32)
    for f in range(e_acts.shape[1]):
        e_pos = e_acts[e_acts[:, f] > 0, f]
        m_pos = m_acts[m_acts[:, f] > 0, f]
        if e_pos.size == 0 or m_pos.size == 0:
            out[f] = np.nan
            continue
        combined = np.concatenate([e_pos, m_pos])
        n_e = e_pos.size
        n_m = m_pos.size
        order = np.argsort(combined)
        z = combined[order]
        is_e = order < n_e
        row = np.arange(1, n_e + n_m + 1)
        cum_e = np.cumsum(is_e)
        fe = cum_e / n_e
        fm = (row - cum_e) / n_m
        dz = np.diff(z)
        out[f] = (np.abs(fe - fm)[:-1] * dz).sum()
    return out


@beartype.beartype
def max_pool_csr(
    ta_csr: scipy.sparse.csr_matrix | scipy.sparse.csr_array, n_images: int, tpi: int
) -> np.ndarray:
    """Max-pool token-level sparse activations to image-level."""
    result = np.zeros((n_images, ta_csr.shape[1]), dtype=np.float32)
    for i in range(n_images):
        s = ta_csr.indptr[i * tpi]
        e = ta_csr.indptr[i * tpi + tpi]
        if s < e:
            np.maximum.at(result[i], ta_csr.indices[s:e], ta_csr.data[s:e])
    return result


@beartype.beartype
def get_scores_fpath(
    run_root_dpath: pathlib.Path, *, run_id: str, shard_id: str
) -> pathlib.Path:
    return run_root_dpath / run_id / "inference" / shard_id / "cambridge-mimics.parquet"


@beartype.beartype
def score_run(
    run_id: str,
    *,
    run_root_dpath: pathlib.Path,
    shard_id: str,
    task_specs: list[TaskSpec],
    n_images: int,
    feature_chunk: int,
    force_recompute: bool,
) -> pl.DataFrame | None:
    pq_fpath = get_scores_fpath(run_root_dpath, run_id=run_id, shard_id=shard_id)
    if pq_fpath.exists() and not force_recompute:
        logger.info("Cached: '%s'.", pq_fpath)
        return pl.read_parquet(pq_fpath)

    ta_fpath = run_root_dpath / run_id / "inference" / shard_id / "token_acts.npz"
    if not ta_fpath.exists():
        logger.warning("Missing token_acts: '%s'.", ta_fpath)
        return None

    ta = scipy.sparse.load_npz(str(ta_fpath))
    d_sae = ta.shape[1]
    tpi = ta.shape[0] // n_images
    assert ta.shape[0] == n_images * tpi, f"{ta.shape[0]} != {n_images} * {tpi}"

    logger.info("Max-pooling %s (shape %s, tpi=%d).", run_id, ta.shape, tpi)
    ma = max_pool_csr(ta, n_images, tpi)
    del ta

    dfs = []
    for spec in task_specs:
        inc = ma[spec.include]
        e_acts = inc[spec.erato_mask]
        m_acts = inc[spec.melp_mask]

        # AUROC via Mann-Whitney U plus Wasserstein-1, chunked for memory.
        auroc = np.empty(d_sae, dtype=np.float32)
        w1_act = np.empty(d_sae, dtype=np.float32)
        w1_firing = np.empty(d_sae, dtype=np.float32)
        for fs in range(0, d_sae, feature_chunk):
            fe = min(fs + feature_chunk, d_sae)
            inc_c = inc[:, fs:fe]
            r = rankdata(inc_c, axis=0)
            ps = r[spec.binary == 1].sum(axis=0)
            auroc[fs:fe] = (ps - spec.n_pos * (spec.n_pos + 1) / 2) / (
                spec.n_pos * spec.n_neg
            )
            e_c = inc_c[spec.erato_mask]
            m_c = inc_c[spec.melp_mask]
            w1_act[fs:fe] = batch_wasserstein1(e_c, m_c)
            w1_firing[fs:fe] = firing_wasserstein1(e_c, m_c)

        dfs.append(
            pl.DataFrame(
                {
                    "run_id": [run_id] * d_sae,
                    "task": [spec.name] * d_sae,
                    "feature_id": np.arange(d_sae, dtype=np.int32),
                    "auroc": auroc,
                    "support_erato": (e_acts > 0).mean(axis=0).astype(np.float32),
                    "support_melpomene": (m_acts > 0).mean(axis=0).astype(np.float32),
                    "mean_act_erato": e_acts.mean(axis=0).astype(np.float32),
                    "mean_act_melpomene": m_acts.mean(axis=0).astype(np.float32),
                    "w1_act": w1_act,
                    "w1_firing": w1_firing,
                }
            )
        )

    del ma
    result = pl.concat(dfs)
    pq_fpath.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(pq_fpath)
    logger.info("Wrote %d rows to '%s'.", result.height, pq_fpath)
    return result


@beartype.beartype
def worker_fn(cfg: Config) -> None:
    """Score all latents for all tasks across multiple runs."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    msg = "Provide at least one run id."
    assert cfg.run_ids, msg

    shards_dpath = cfg.shards_dpath.expanduser()
    msg = f"Shards missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg

    _, labels_by_col = load_image_labels(shards_dpath)
    sv = labels_by_col["subspecies_view"]
    n_images = len(sv)
    shard_id = shards_dpath.name

    task_specs = build_task_specs(
        sv, pair_specs=cfg.pair_specs, views=cfg.views, min_samples=cfg.min_samples
    )
    logger.info(
        "Built %d task specs from %d pairs x %d views.",
        len(task_specs),
        len(cfg.pair_specs),
        len(cfg.views),
    )

    for i, run_id in enumerate(cfg.run_ids, start=1):
        logger.info("[%d/%d] Scoring run '%s'.", i, len(cfg.run_ids), run_id)
        score_run(
            run_id,
            run_root_dpath=cfg.run_root_dpath,
            shard_id=shard_id,
            task_specs=task_specs,
            n_images=n_images,
            feature_chunk=cfg.feature_chunk,
            force_recompute=cfg.force_recompute,
        )

    logger.info("Done scoring %d runs.", len(cfg.run_ids))
