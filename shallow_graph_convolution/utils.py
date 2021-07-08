#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import os
from logging import getLogger


def get_checkpoint_dir():
    """Get the checkpoint dir"""
    logger = getLogger()
    checkpoint_dir_path = os.getenv(
        "RECBOLE_CHECKPOINT_DIR",
        os.path.expanduser(os.path.join("~", "recbole_checkpoint_dir")),
    )
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
        logger.info(f"Checkpoint directory created at: {checkpoint_dir_path}")
    return checkpoint_dir_path
