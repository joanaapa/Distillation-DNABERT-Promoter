# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

""" Fine-tuning the library models for sequence classification. 
    Prepared to fine-tune for promoter identification.
"""


import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from multiprocessing import Pool
from typing import Dict, List, Tuple

import neptune.new as neptune
import numpy as np
import torch
from decouple import config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from data_loaders import load_and_cache_examples_promoter as load_and_cache_examples
from data_loaders import visualize 


from src.transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DNATokenizer,
    get_linear_schedule_with_warmup,
)
from src.transformers import glue_compute_metrics as compute_metrics
from src.transformers import glue_output_modes as output_modes
from src.transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            DistilBertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "dnaprom": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "distildnaprom": (DistilBertConfig, DistilBertForSequenceClassification, DNATokenizer),
    "minidnaprom": (BertConfig, BertForSequenceClassification, DNATokenizer),
}

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    """
    It takes a list of checkpoint files, and returns a list of checkpoint files sorted by their epoch
    number
    
    :param args: The arguments that were passed to the script
    :param checkpoint_prefix: The prefix of the checkpoint files, defaults to checkpoint (optional)
    :param use_mtime: If True, the checkpoint with the latest modification time will be used. If False,
    the checkpoint with the highest number will be used, defaults to False (optional)
    :return: A list of strings.
    """
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    """
    It deletes the oldest checkpoints if the number of checkpoints exceeds the `save_total_limit`
    argument
    
    :param args: The arguments object that was passed to the training script
    :param checkpoint_prefix: The prefix of the checkpoint files, defaults to checkpoint (optional)
    :param use_mtime: If True, the checkpoint with the latest modification time will be returned. If
    False, the checkpoint with the latest step number will be returned, defaults to False (optional)
    :return: the list of checkpoints sorted by their creation time.
    """
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer, run=None):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1, args.beta2)
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # multi-gpu training 
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training 
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    
    if args.do_visualize_during_training:
        tata_dataset = load_and_cache_examples(args, args.task_name, tokenizer, viz=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = 0
    last_auc = 0
    stop_count = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if run is not None:
                run["train/lr"].log(scheduler.get_lr()[0], step=global_step)
                run["train/loss"].log(loss, step=global_step)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                if args.local_rank in [-1, 0] and args.do_visualize_during_training and global_step % args.image_steps == 0:
                    kmer = int(args.tokenizer_name[-1])
                    attention_scores, _ = visualize(args, model, tokenizer, prefix="", kmer=kmer, pred_dataset=tata_dataset)
                    #Plot
                    fig, ax = plt.subplots(figsize=(6,3))
                    sns.set()
                    sns.set(font_scale=1.2)
                    ax = sns.heatmap(attention_scores, cmap='YlGnBu', vmin=0, ax=ax, yticklabels=1500) #xticklabels=bases
                    plt.xticks(np.arange(0,300,25), np.arange(-250,50,25), rotation=45, fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.tight_layout()
                    ax.set_xlabel('Coordinate', fontsize=14)
                    ax.set_ylabel('Sequence', fontsize=14)
                    if run is not None:
                        run["train/attention"].log(fig, step=global_step)
                    

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, evaluate=False, val=True)
                        if run is not None:
                            run["val/acc"].log(results["acc"], step=global_step)
                            run["val/f1"].log(results["f1"], step=global_step)
                            run["val/mcc"].log(results["mcc"], step=global_step)
                            run["val/auc"].log(results["auc"], step=global_step)
                            run["val/precision"].log(results["precision"], step=global_step)
                            run["val/recall"].log(results["recall"], step=global_step)
                        

                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

                # Save Checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", evaluate=True, val=False):
    eval_task = args.task_name
    eval_output_dir = args.output_dir
    softmax = torch.nn.Softmax(dim=1)

    results = {}
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate, val=val)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    probs = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(eval_task, preds, out_label_ids, probs)
    results.update(result)

    # Save results in a txt file
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        eval_result = args.data_dir.split("/")[-1] + " "

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            eval_result = eval_result + str(result[key])[:5] + " "
        writer.write(eval_result + "\n")

    return results


def predict(args, model, tokenizer, prefix=""):
    pred_task = args.task_name
    pred_output_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    softmax = torch.nn.Softmax(dim=1)

    predictions = {}
    pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.pred_batch_size)
    pred_loss = 0.0
    nb_pred_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(pred_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(pred_task, preds, out_label_ids, probs)

    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedir(pred_output_dir)
    output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")
    logger.info("***** Pred results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    np.save(output_pred_file, probs)



def main():
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--task_name", default='dnaprom', type=str, help="Script only prepared for promoter task" )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--tokenizer_name",default="dna6",type=str, help="Pretrained tokenizer name or path if not the same as model_name",)

    # OBJECTIVE
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")

    # VALIDATION DURING TRAINING / EVALUATE 
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",)
    parser.add_argument("--do_visualize_during_training", action="store_true", help="Steps to generate an image")
    parser.add_argument("--image_steps", type=int, default=0, help="Steps to generate an image")
    
    # MODEL CONFIGS (only use)

    # TRAINING DETAILS
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument("--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--n_process", default=2, type=int, help="number of processes used for data process",)
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",)
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--neptune", action="store_true", help="Neptune")
    parser.add_argument("--neptune_tags", type=list, default=["trial"], help="Neptune tags")
    parser.add_argument("--neptune_description", type=str, default="TRIAL minilm fine-tuning", help="Neptune description")
    parser.add_argument("--neptune_token", type=str, default=None, help="Neptune API token")
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project")
    


    # OTHER
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)


    args = parser.parse_args()


    if args.do_train and args.neptune:
        run = neptune.init(
            project=args.neptune_project,
            api_token=args.neptune_token,
            tags=args.neptune_tags,
            description=args.neptune_description,
        )
        run["sys/tags"].add([args.task_name, args.tokenizer_name[-1] + "kmer"])

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
    # Segons els que he entès, local_rank és per si utilitzes més d'una màquina. Pot ser que utilitzis 1+ gpu però totes a la mateixa màquina
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print("devices", device)
        print("Number of gpus", args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]  # 'classification' or 'regressions'
    label_list = processor.get_labels()  # [0,1]
    num_labels = len(label_list)


    # LOAD AND INITIALIZE MODELS -----------------------------------------------------------------------------------------------------

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    
    if not args.do_visualize:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        config.output_attentions=True
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.split = int(args.max_seq_length / 512)
        

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        logger.info("finish loading model")

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)


    # TRAIN (for promoter) -----------------------------------------------------------------------------------------------------
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if not args.neptune:
            run = None
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, run)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        if args.neptune:
            run["model/parameters"] = vars(args)

        # Load a trained model and vocabulary that you have fine-tuned to continue for evaluation
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # EVALUATION ON THE TEST SET (for promoter) -----------------------------------------------------------------------------------------------------
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.model_name_or_path]
        # To evaluate all checkpoints in the folder
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.train()
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix) # Results saved in file eval_results.txt
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
        print(results)

        # Neptune log final evaluation
        if args.do_train and args.neptune:
            run["test/acc"] = results["acc_"]
            run["test/f1"] = results["f1_"]
            run["test/mcc"] = results["mcc_"]
            run["test/auc"] = results["auc_"]
            run["test/precision"] = results["precision_"]
            run["test/recall"] = results["recall_"]


    # PREDICTION OF THE TEST SET (for promoter) + EVALUATE -----------------------------------------------------------------------------------------
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)
        prediction = predict(args, model, tokenizer)


    # ATTENTION VISUALIZATION (TATA promoters) -----------------------------------------------------------------------------------------------------
    if args.do_visualize and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)

        tata_dataset = load_and_cache_examples(args, args.task_name, tokenizer, viz=True)   
        kmer = int(args.tokenizer_name[-1])
        attention_scores, _ = visualize(args, model, tokenizer, prefix="", kmer=kmer, pred_dataset=tata_dataset)

        #Plot
        fig, ax = plt.subplots(figsize=(6,3))
        sns.set()
        sns.set(font_scale=1.2)
        ax = sns.heatmap(attention_scores, cmap='YlGnBu', vmin=0, ax=ax, yticklabels=1500) #xticklabels=bases
        plt.xticks(np.arange(0,300,25), np.arange(-250,50,25), rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        ax.set_xlabel('Coordinate', fontsize=14)
        ax.set_ylabel('Sequence', fontsize=14)

        plt.savefig(os.path.join(args.output_dir, "atten_viz.png"))
        np.save(os.path.join(args.output_dir, "atten.npy"), attention_scores)

    return results


if __name__ == "__main__":
    main()
