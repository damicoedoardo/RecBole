#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import argparse

from shallow_graph_convolution.wandb_load_model import laod_wandb_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train model")
    parser.add_argument(
        "--model", "-m", type=str, help="model name", default="LightGCN"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, help="dataset name", default="ml-100k"
    )
    parser.add_argument("--loss_type", "-lt", type=str, default="BPR")
    parser.add_argument(
        "--config_file_list",
        "-cfl",
        type=str,
        default=["dataset", "environment", "evaluation", "training"],
    )
    args, _ = parser.parse_known_args()

    # load model
    model = laod_wandb_model(args)

    #wandb_trainer = WandbTrainer(config, model)
