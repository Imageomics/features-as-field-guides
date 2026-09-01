# Automatic Field Guides: Sparse Autoencoders and Heuristic Ranking for Dichotomous Key Learning

![MIT License](https://img.shields.io/badge/License-MIT-efefef)

A fork of [saev](imageomics.github.io/saev/), modified for dichotomous key recovery and discovery in images of Heliconius butterfly mimics. 

## Using this Repository

All scripts for training SAEs, generating sparse features, scoring features for selectivity and W1 distance, and viewing dichotomous key candidates are located in `contrib/mimics/exps/001-features/`

First, generate ViT patch embedding shards using `generate_dino_shards.sh`. Then, train SAEs using `train_dino_saes.sh` and generate sparse features using `inference_dino.sh`. Finally, score features by running the following command:

```bash
uv run contrib/mimics/launch.py score
```

Visualization can be done using `uv run marimo edit contrib/mimics/exps/001-features/dino_viewer.py`.

## Citation

If you want to cite the software, please use the "Cite this repository" link on the GitHub page, which will provide you with the appropriate citation format.
