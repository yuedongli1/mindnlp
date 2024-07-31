# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Callback for saving checkpoint.
"""
import os
from pathlib import Path
import mindspore
from mindnlp._legacy.abc import Callback


class CheckpointCallback(Callback):
    """
    Save checkpoint of the model. save the current Trainer state at the end of each epoch, which can be used to
    resume previous operations.
    Continue training a sample code using the most recent epoch

    Args:
        save_path (str, Path): The path to save the state. A specific path needs to be specified,
            such as 'checkpoints/'.
        ckpt_name (str): Checkpoint name to store. It will set model class name when not specified.
            Default: None.
        epochs (int): Save a checkpoint file every n epochs.
        keep_checkpoint_max (int): Save checkpoint files at most. Default:5.

    """
    def __init__(self, save_path, ckpt_name=None, epochs=None, keep_checkpoint_max=5):
        if isinstance(save_path, str):
            self.save_path = Path(save_path)
        elif isinstance(save_path, Path):
            self.save_path = save_path
        else:
            raise ValueError(f"the 'save_path' argument must be str or Path, but got {type(save_path)}.")

        if not self.save_path.exists():
            os.makedirs(str(self.save_path))

        self.epochs = epochs
        self.keep_checkpoint_max = keep_checkpoint_max
        self.ckpt_name = ckpt_name
        self.cached_ckpts = []

        # to do

        # self.steps = steps
        # if (self.epochs is not None) & (self.steps is not None):
        #     raise ValueError("The parameter epochs and steps cannot be assigned at the same time,\
        #                         you can only keep one of them.")
        # elif (self.epochs is None) & (self.steps is None):
        #     raise ValueError("The parameter epochs and steps both are None,\
        #                         you must assign one of them.")

    def train_begin(self, run_context):
        """
        Notice the file saved path of checkpoints at the beginning of training.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.epochs is None:
            raise ValueError('For saving checkpoints, epoch cannont be `None` !')
        print(f"The train will start from the checkpoint saved in '{self.save_path}'.")

    def train_epoch_end(self, run_context):
        """
        Save checkpoint every n epochs at the end of the epoch.

        Args:
            run_context (RunContext): Information about the model.

        """
        if self.epochs is None:
            return
        if (run_context.cur_epoch_nums % self.epochs != 0) & (run_context.cur_epoch_nums != run_context.epochs):
            return
        model = run_context.network
        if self.ckpt_name is None:
            self.ckpt_name = type(model).__name__
        ckpt_name = self.ckpt_name + '_epoch_' + str(run_context.cur_epoch_nums-1) + '.ckpt'
        if len(self.cached_ckpts) == self.keep_checkpoint_max:
            print('The maximum number of stored checkpoints has been reached.')
            del_ckpt = self.cached_ckpts.pop(0)
            del_file = self.save_path.joinpath(del_ckpt)
            del_file.chmod(0o777)
            del_file.unlink()

        mindspore.save_checkpoint(model, str(self.save_path.joinpath(ckpt_name).resolve()))
        self.cached_ckpts.append(ckpt_name)
        print(f"Checkpoint: '{ckpt_name}' has been saved in epoch: {run_context.cur_epoch_nums - 1}.")
