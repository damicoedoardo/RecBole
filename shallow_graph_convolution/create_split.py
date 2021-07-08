#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import logging
from logging import getLogger

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=str, help="name of dataset", default="ml-100k"
    )
    args, _ = parser.parse_known_args()

    config_dict = {"loss_type": "BPR", "checkpoint_dir": "../saved"}

    BASE_YAML_PATH = "config_yaml_files/"
    config = Config(
        model=GeneralRecommender,
        dataset="ml-100k",
        config_dict=config_dict,
        config_file_list=[
            BASE_YAML_PATH + "environment.yaml",
            BASE_YAML_PATH + "training.yaml",
            BASE_YAML_PATH + "evaluation.yaml",
            BASE_YAML_PATH + "dataset.yaml",
        ],
    )
    # initialise all the seeds related
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    # dataset filtering
    dataset = create_dataset(config)

    init_logger(config)
    logger = getLogger()
    logger.info(dataset)

    # save filtered dataset
    dataset.save("../saved/")

    train_data, valid_data, test_data = data_preparation(config, dataset)
    save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))
