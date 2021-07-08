#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"


def get_best_model_path():
    """Return the best model checkpoint path if needed the file is downloaded from wandb"""


def get_wandb_project_dict(dataset: str):
    """ Retrieve run id for the best model over the dataset"""
    w_project_dict = \
        {
            "ml-100k": {
                "project_name": 'RecBole-shallow_graph_convolution_models_LightGCN',
                "LightGCN": 'kb3f7xso',
                "BPR": ''
            }
        }
    return w_project_dict[dataset]
