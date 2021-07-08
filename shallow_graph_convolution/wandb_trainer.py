#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import os
from pathlib import Path
from time import time

import torch
import wandb

from recbole.trainer import Trainer
from recbole.utils import (
    get_local_time,
    early_stopping,
    dict2str,
)
from recbole.utils.utils import set_color
from shallow_graph_convolution.utils import get_checkpoint_dir


class WandbTrainer(Trainer):
    def __init__(self, config, model):
        super(WandbTrainer, self).__init__(config, model)
        self.model_file = "{}-{}.pth".format(self.config["model"], get_local_time())

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        # initialize wandb and the save directory where to store the model checkpoint
        # todo: not clean way of defining save path it should be inside init...
        wandb.init(
            config=self.config.final_config_dict, resume=True, dir=get_checkpoint_dir()
        )
        # overwrite save dir so that it will uploaded to wandb
        self.saved_model_file = os.path.join(wandb.run.dir, self.model_file)

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            parsed_train_loss = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            self.train_loss_dict[epoch_idx] = parsed_train_loss
            # log loss on wandb
            wandb.log({"loss": parsed_train_loss}, step=epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = (
                        set_color("Saving current", "blue")
                        + ": %s" % self.saved_model_file
                    )
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                # log on wandb valid results
                wandb.log(valid_result, step=epoch_idx)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = (
                            set_color("Saving current best", "blue")
                            + ": %s" % self.saved_model_file
                        )
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break
        if self.draw_loss_pic:
            save_path = "{}-{}-train_loss.pdf".format(
                self.config["model"], get_local_time()
            )
            self.plot_train_loss(save_path=os.path.join(save_path))
        return self.best_valid_score, self.best_valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)
