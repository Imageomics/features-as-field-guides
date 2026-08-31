"""
Baseline image-level classifiers for Cambridge mimic pairs.

Measures how well a simple logistic-regression head separates erato from
melpomene for each subspecies pair task, using the [CLS] embedding read
directly from recorded activation shards. No forward pass happens here; the
user supplies a shards dir with recorded activations for either model.
"""

import dataclasses
import logging
import pathlib

import beartype
import numpy as np
import polars as pl
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from jaxtyping import Float, jaxtyped
from tdiscovery.classification import load_image_labels

from saev.data import Metadata
from saev.data.shards import ShardInfo

from . import tasks
from .scoring import TaskSpec, build_task_specs

logger = logging.getLogger(__name__)

DEFAULT_OUT_DPATH = pathlib.Path("/fs/ess/PAS2136/jbeattie/saev/baselines")

METRICS: tuple[str, ...] = ("bal_acc", "acc", "precision", "recall", "auc")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Measure a classifier head on model embeddings for mimic pair tasks."""

    shards_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/jbeattie/saev/shards/46bcd3db"
    )
    """Shards dir with recorded activations for either model. Set via CLI or sweep."""
    layer: int | None = None
    """Layer to read. None -> deepest recorded layer (final embedding)."""
    pair_specs: list[tuple[str, str]] = dataclasses.field(
        default_factory=lambda: [
            tuple(tasks.parse_pair_spec(spec)) for spec in tasks.DEFAULT_PAIR_SPECS
        ]
    )
    """(erato_ssp, melpomene_ssp) pairs to evaluate."""
    views: list[str] = dataclasses.field(
        default_factory=lambda: tasks.DEFAULT_VIEWS.copy()
    )
    """Views to evaluate."""
    min_samples: int = 50
    """Minimum images per class to include a task."""
    n_splits: int = 5
    """Number of stratified CV folds."""
    seed: int = 42
    """Random seed for the CV split."""
    force_recompute: bool = False
    """If True, recompute even when the output parquet already exists."""
    out_fpath: pathlib.Path | None = None
    """Output parquet path. Defaults to baselines/baseline_cls_<shard>_layer<layer>.parquet."""

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


@jaxtyped(typechecker=beartype.beartype)
def load_cls_embeddings(
    shards_dpath: pathlib.Path, *, layer: int
) -> Float[np.ndarray, "n_images d_model"]:
    """Read the [CLS] token embedding for every image at the given layer.

    Streams each shard through a memmap and slices out the single [CLS] row,
    so peak memory is bounded by one shard's [CLS] slice, not the full shard.
    """
    md = Metadata.load(shards_dpath)
    msg = "Shards must record a [CLS] token to extract an image embedding."
    assert md.cls_token, msg
    msg = f"Layer {layer} not recorded in {md.layers}."
    assert layer in md.layers, msg

    shard_info = ShardInfo.load(shards_dpath)
    shard_info.validate(shards_dpath)

    layer_idx = md.layers.index(layer)
    emb = np.empty((md.n_examples, md.d_model), dtype=np.float32)
    start = 0
    for shard in shard_info:
        acts = np.memmap(
            shards_dpath / shard.name, mode="r", dtype=np.float32, shape=md.shard_shape
        )
        emb[start : start + shard.n_examples] = acts[
            : shard.n_examples, layer_idx, 0, :
        ]
        start += shard.n_examples
        del acts

    msg = f"Read {start} embeddings, expected {md.n_examples}."
    assert start == md.n_examples, msg
    return emb


@jaxtyped(typechecker=beartype.beartype)
def eval_task_cv(
    emb: Float[np.ndarray, "n_images d_model"],
    spec: TaskSpec,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Stratified CV metrics for one task.

    Returns mean/std per metric across folds. precision and recall are
    macro-averaged over the two classes (symmetric with bal_acc); auc is
    ROC AUC on the melpomene-vs-erato scores.
    """
    X = emb[spec.include]
    y = spec.binary
    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
    fold_values: dict[str, list[float]] = {name: [] for name in METRICS}
    for train_i, test_i in skf.split(X, y):
        clf.fit(X[train_i], y[train_i])
        pred = clf.predict(X[test_i])
        fold_values["bal_acc"].append(
            sklearn.metrics.balanced_accuracy_score(y[test_i], pred)
        )
        fold_values["acc"].append(sklearn.metrics.accuracy_score(y[test_i], pred))
        fold_values["precision"].append(
            sklearn.metrics.precision_score(y[test_i], pred, average="macro")
        )
        fold_values["recall"].append(
            sklearn.metrics.recall_score(y[test_i], pred, average="macro")
        )
        fold_values["auc"].append(
            sklearn.metrics.roc_auc_score(y[test_i], clf.predict_proba(X[test_i])[:, 1])
        )
    return {
        name: (float(np.mean(vals)), float(np.std(vals)))
        for name, vals in fold_values.items()
    }


@beartype.beartype
def get_empty_df() -> pl.DataFrame:
    schema: dict[str, type | pl.DataType | None] = {
        "shard_id": pl.String,
        "family": pl.String,
        "ckpt": pl.String,
        "layer": pl.Int64,
        "token": pl.String,
        "task_name": pl.String,
        "n_erato": pl.Int64,
        "n_melpomene": pl.Int64,
        "n_splits": pl.Int64,
    }
    for name in METRICS:
        schema[f"{name}_mean"] = pl.Float64
        schema[f"{name}_std"] = pl.Float64
    return pl.DataFrame(schema=schema)


@beartype.beartype
def worker_fn(cfg: Config) -> None:
    """Evaluate baseline classifier heads for all mimic pair tasks."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    shards_dpath = cfg.shards_dpath.expanduser()
    msg = f"Shards directory missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg

    md = Metadata.load(shards_dpath)
    layer = cfg.layer if cfg.layer is not None else max(md.layers)
    shard_id = shards_dpath.name

    out_fpath = cfg.out_fpath or (
        DEFAULT_OUT_DPATH / f"baseline_cls_{shard_id}_layer{layer}.parquet"
    )
    if out_fpath.exists() and not cfg.force_recompute:
        logger.info("Cached: '%s'.", out_fpath)
        return

    logger.info("Reading [CLS] embeddings at layer %d from '%s'.", layer, shards_dpath)
    emb = load_cls_embeddings(shards_dpath, layer=layer)

    _, labels_by_col = load_image_labels(shards_dpath)
    sv = labels_by_col["subspecies_view"]
    n_images = len(sv)
    msg = f"Label count {n_images} != shard examples {md.n_examples}."
    assert n_images == md.n_examples, msg

    task_specs = build_task_specs(
        sv, pair_specs=cfg.pair_specs, views=cfg.views, min_samples=cfg.min_samples
    )
    logger.info("Built %d task specs.", len(task_specs))

    rows = []
    for spec in task_specs:
        stats = eval_task_cv(emb, spec, n_splits=cfg.n_splits, seed=cfg.seed)
        row = {
            "shard_id": shard_id,
            "family": md.family,
            "ckpt": md.ckpt,
            "layer": layer,
            "token": "cls",
            "task_name": spec.name,
            "n_erato": spec.n_neg,
            "n_melpomene": spec.n_pos,
            "n_splits": cfg.n_splits,
        }
        for name in METRICS:
            mean, std = stats[name]
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        rows.append(row)
        stats_str = ", ".join(
            f"{name}={stats[name][0]:.4f}+-{stats[name][1]:.4f}" for name in METRICS
        )
        logger.info(
            "[%d/%d] %s: %s (n_erato=%d, n_melpomene=%d).",
            len(rows),
            len(task_specs),
            spec.name,
            stats_str,
            spec.n_neg,
            spec.n_pos,
        )

    result = pl.DataFrame(rows, infer_schema_length=None) if rows else get_empty_df()
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_fpath)
    logger.info("Wrote %d rows to '%s'.", result.height, out_fpath)
