#!/usr/bin/env python
"""Train and evaluate LPAE-Net (CVPR 2021) on outfit recommendation datasets.

Datasets are downloaded from Hugging Face into ``data/`` -- see README.md.

Examples::

    # Train LPAE-u on Polyvore-630
    python main.py train \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn --gpus 0

    # Evaluate AUC / NDCG with random negatives
    python main.py evaluate \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model_195_val_auc=0.8935.pt

    # Evaluate Fill-In-The-Blank accuracy
    python main.py fitb \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model_195_val_auc=0.8935.pt
"""

import argparse

import yaml

from runner import Evaluator, Trainer, build_net, load_trained
from runner.utils import get_logger, load_yaml, save_yaml, select_device, set_seed, setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="LPAE-Net", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["train", "evaluate", "fitb"], help="action to run")
    parser.add_argument("--cfg", required=True, help="configuration file")
    parser.add_argument("--name", default=None, help="name for the log files")
    parser.add_argument("--log-dir", default=None, help="folder for logs and checkpoints")
    parser.add_argument("--load-trained", default=None, help="checkpoint to evaluate")
    parser.add_argument("--gpus", default=None, help="comma separated GPU ids, e.g. 0")
    parser.add_argument("--num-runs", type=int, default=1, help="number of evaluation runs")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg)

    name = args.name or args.action
    log_dir = args.log_dir or cfg.get("log_dir", "summaries/debug")
    log_dir = setup_logging(log_dir, name)
    set_seed(cfg.get("seed"))
    device = select_device(args.gpus)

    logger = get_logger("main")
    logger.info("Configuration:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    save_yaml(cfg, f"{log_dir}/{name}.yaml")

    if args.action == "train":
        best_checkpoint = Trainer(cfg, device, log_dir).fit()
        net = build_net(cfg, device)
        load_trained(net, best_checkpoint)
        logger.info("Evaluating the best model on the test split.")
        Evaluator(cfg, device).evaluate(net, num_runs=args.num_runs)
    else:
        assert args.load_trained, "--load-trained is required for evaluate / fitb"
        net = build_net(cfg, device)
        load_trained(net, args.load_trained)
        evaluator = Evaluator(cfg, device)
        if args.action == "evaluate":
            evaluator.evaluate(net, num_runs=args.num_runs)
        else:
            evaluator.fitb(net, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
