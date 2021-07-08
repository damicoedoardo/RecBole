#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import os

from shallow_graph_convolution.train_model import train_model_wandb
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train model")
    parser.add_argument("--model", "-m", type=str, help="model name", default="SGCCF")
    parser.add_argument(
        "--dataset", "-d", type=str, help="dataset name", default="ml-100k"
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, help="learning rate", default=1e-2
    )
    parser.add_argument(
        "--embedding_size", "-es", type=int, help="size of the embedding", default=64
    )
    parser.add_argument(
        "--n_layers", "-nl", type=int, help="convolution depth", default=3
    )
    parser.add_argument(
        "--propagation_matrix", type=str, help="Propagation matrix kind", default="SNL"
    )
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        help="weight for l2 regularization",
        default=0.0,
    )
    parser.add_argument("--loss_type", "-lt", type=str, default="BPR")
    parser.add_argument(
        "--config_file_list",
        "-cfl",
        type=str,
        default=["dataset", "environment", "evaluation", "training"],
    )
    args, _ = parser.parse_known_args()
    train_model_wandb(vars(args))
