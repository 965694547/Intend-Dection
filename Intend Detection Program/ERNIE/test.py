# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import os
import random
import time
import distutils.util
import pandas as pd

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from utils import convert_example
from data_loader_fn import data_loader_process

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--task", default='dh_nh', required=False, type=str, help="The name of the task to train")
parser.add_argument("--params_path", type=str, required=False, default="checkpoints/dh_nh/model_100000/model_state.pdparams", help="The path to model parameters to be loaded.")

parser.add_argument("--save_dir", default='./checkpoint/dh_nh', type=str, help="The output directory where the model checkpoints will be written.")
#parser.add_argument("--dataset", choices=["chnsenticorp", "xnli_cn"], default="chnsenticorp", type=str, help="Dataset for classfication tasks.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=20, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--valid_steps", default=5000, type=int, help="The interval steps to evaluate model performance.")
parser.add_argument("--save_steps", default=5000, type=int, help="The interval steps to save checkppoints.")
parser.add_argument("--logging_steps", default=5000, type=int, help="The interval steps to logging.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--use_amp", type=distutils.util.strtobool, default=False, help="Enable mixed precision training.")
parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    predict = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        predict.extend(logits.argmax(1).tolist())
        metric.update(correct)
    accu = metric.accumulate()
    print("eval loss: %.5f, accuracy: %.5f" % (np.mean(losses), accu))
    output = pd.DataFrame(data=predict)
    output.to_csv('error.csv',index=None,header=None)
    model.train()
    metric.reset()


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    #train_ds, dev_ds, test_ds = load_dataset(
    #    args.dataset, splits=["train", "dev", "test"])
    train_ds_copy, train_ds, dev_ds, test_ds=data_loader_process(args.task)

    model_name='ernie-2.0-en'
    if args.task == 'dh_nh':
        model_name='ernie-1.0'

    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        model_name, num_classes=len(train_ds.label_list))
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(model_name)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_pair=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    print("Test:")
    test_data_loader = create_dataloader(
        test_ds,
        mode='dev',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    evaluate(model, criterion, metric, test_data_loader)


if __name__ == "__main__":
    do_train()
