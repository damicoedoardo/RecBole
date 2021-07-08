#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import wandb
import os
import wandb
import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import load_split_dataloaders, set_color
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from shallow_graph_convolution.best_models import get_wandb_project_dict
from shallow_graph_convolution.utils import get_checkpoint_dir
from shallow_graph_convolution.wandb_trainer import WandbTrainer
import wandb


def laod_wandb_model(args):
    """Load model from wandb server"""
    # create correct path to config files
    # todo: move base path to config file
    BASE_YAML_PATH = "shallow_graph_convolution/config_yaml_files/"
    cfl = list(map(lambda x: BASE_YAML_PATH + x + ".yaml", args.config_file_list))
    # configurations initialization
    config = Config(model=args.model, dataset=args.dataset, config_file_list=cfl,)
    init_seed(config["seed"], config["reproducibility"])

    pdict = get_wandb_project_dict(args.dataset)
    project_name = pdict["project_name"]
    run_id = pdict[args.model]

    # load splitted data
    # todo: load the correct dataset
    train_data, valid_data, test_data = load_split_dataloaders(
        "saved/ml-100k-for-GeneralRecommender-dataloader.pth"
    )
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # model loading and initialization
    model = get_model(config["model"])(config, train_data).to(config["device"])

    model_checkpoint_path = wandb.restore(
        "LightGCN-Jul-07-2021_16-33-38.pth", run_path=f"damicoedoardo/{project_name}/{run_id}", root=get_checkpoint_dir()
    )

    checkpoint = torch.load(
        model_checkpoint_path.name
    )
    model.load_state_dict(checkpoint["state_dict"])

    logger.info('Restored model:\n')
    logger.info(model)
    return model
