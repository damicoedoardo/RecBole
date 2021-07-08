#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import os
import wandb
import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import load_split_dataloaders, set_color
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from shallow_graph_convolution.utils import get_checkpoint_dir
from shallow_graph_convolution.wandb_trainer import WandbTrainer


def train_model_wandb(config_dict):
    """Generic train routine for a model"""
    # set checkpoint_dir
    checkpoint_dir = get_checkpoint_dir()
    config_dict.update({"checkpoint_dir": checkpoint_dir})

    # create correct path to config files
    BASE_YAML_PATH = "shallow_graph_convolution/config_yaml_files/"
    cfl = list(map(lambda x: BASE_YAML_PATH + x + ".yaml", config_dict['config_file_list']))
    # configurations initialization
    config = Config(
        model=config_dict['model'],
        dataset=config_dict['dataset'],
        config_file_list=cfl,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

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
    logger.info(model)

    # trainer loading and initialization
    # trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # init wandb
    wandb_trainer = WandbTrainer(config, model)

    # model training
    wandb_trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = wandb_trainer.evaluate(test_data, load_best_model=True)
    logger.info(set_color("test result", "yellow") + f": {test_result}")
