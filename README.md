# LPAE-Net

This repository contains the code of the paper _"Personalized Outfit Recommendation with Learnable Anchors"_ (CVPR 2021).

The code is self-contained: it only depends on PyTorch and a few common Python packages, and reads the preprocessed datasets published on [Hugging Face](https://huggingface.co/datasets/lzcn/outfit-datasets).

## Requirements

```bash
pip install -r requirements.txt
```

## Data preparation

Download the preprocessed datasets (pre-extracted ResNet-34 item features + outfit tuples) from Hugging Face and place them under `data/` in the repository root:

```bash
git lfs install
git clone https://huggingface.co/datasets/lzcn/outfit-datasets data
```

The layout used by this project:

```text
data/
└── polyvore-u/
    ├── features/
    │   └── resnet34/           # pre-extracted ResNet-34 item features (LMDB)
    └── original/
        ├── tuples_630/         # 630-user personalized split (+ FITB questions)
        └── tuples_519/         # 519-user personalized split
```

> _ResNet-34-nn_ means the pretrained image features extracted from [ResNet-34](https://arxiv.org/abs/1512.03385) are used directly, i.e. the backbone is not fine-tuned and no raw images are needed.

## Quick check

Run a 1-epoch sanity-check training (~1 minute on a GPU):

```bash
python run_lpae_net.py train --cfg configs/smoke_test.yaml --log-dir summaries/smoke_test
```

## Train

```bash
python run_lpae_net.py train \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
    --gpus 0 \
    --name train
```

Logs are written to `summaries/<log-dir>/train.log`, TensorBoard events are recorded alongside, and the top-5 checkpoints ranked by validation AUC are kept under `summaries/<log-dir>/checkpoints/`.

## Evaluate

AUC / NDCG with randomly mixed negatives (`test.neg_ratio`, averaged over `--num-runs`):

```bash
python run_lpae_net.py evaluate \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
    --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model_XXX_val_auc=0.XXXX.pt \
    --gpus 0 \
    --num-runs 10 \
    --name evaluate-auc
```

Fill-In-The-Blank accuracy using the pre-computed questions shipped with the dataset:

```bash
python run_lpae_net.py fitb \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
    --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model_XXX_val_auc=0.XXXX.pt \
    --gpus 0 \
    --name evaluate-fitb
```

To run on Polyvore-_519_ or with the shared-anchor variant, swap the config for `configs/polyvore_519_*` / `*_lpae_g_*`.

## Configuration

Each YAML file in `configs/` is fully self-contained:

| Key | Description |
| --- | --- |
| `data.root` / `data.features` | split folder and feature reader path |
| `train/valid/test.neg_ratio` | negatives per positive; `<= 0` uses the fixed `{phase}_neg` files |
| `net.name` | `lpae_u`, `lpae_g`, or `LatentFactorNet` |
| `optim` | optimizer + LR scheduler (stepped on validation AUC) |

## Logs

Training logs released with the paper:

- LPAE-_u_ (_ResNet-34-nn_) Polyvore-_630_: [config](configs/polyvore_630_lpae_u_resnet34_nn.yaml) · [train.log](summaries/polyvore_630_lpae_u_resnet34_nn/train.log)
- LPAE-_u_ (_ResNet-34-nn_) Polyvore-_519_: [config](configs/polyvore_519_lpae_u_resnet34_nn.yaml) · [train.log](summaries/polyvore_519_lpae_u_resnet34_nn/train.log)

## Contact

email: zhilu@std.uestc.edu.cn
