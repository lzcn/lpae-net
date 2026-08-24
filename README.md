# LPAE-Net

Code of _"Personalized Outfit Recommendation with Learnable Anchors"_ (CVPR 2021).

## Setup

```bash
pip install -r requirements.txt
git lfs install && git clone https://huggingface.co/datasets/lzcn/outfit-datasets data
```

Expects `data/polyvore-u/original/tuples_{630,519}` and `data/polyvore-u/features/resnet34`.

## Train

```bash
python main.py train \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --log-dir summaries/polyvore_630_lpae_u \
    --gpus 0
```

## Evaluate

```bash
# AUC / NDCG (random negatives)
python main.py evaluate \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --load-trained <checkpoint> \
    --num-runs 10

# Fill-In-The-Blank
python main.py fitb \
    --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
    --load-trained <checkpoint>
```

Configs are self-contained; `polyvore_519_*` and `*_lpae_g_*` swap dataset or variant, `test.neg_ratio <= 0` switches to fixed negatives.

## Logs

Training logs of the paper: [Polyvore-630](summaries/polyvore_630_lpae_u_resnet34_nn/train.log) · [Polyvore-519](summaries/polyvore_519_lpae_u_resnet34_nn/train.log)

> Refactored with an LLM.

## License

MIT.
