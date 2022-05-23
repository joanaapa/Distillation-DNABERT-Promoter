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

"""
DistilBERT and MiniLM distillation, evaluation and promoter visualization.
"""


import argparse
import glob
import os
import re
import random
import sys
from typing import List

import numpy as np
import torch
from utils import logger

from distiller import Distiller
from initialise_distilbert import init_student
from data_loaders import *


from src.transformers import (
    DNATokenizer,
    MiniLMForPreTraining,
)
from src.transformers import BertConfig as MiniLMConfig

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)



MODEL_CLASSES = {
    "dna": (BertConfig, BertForMaskedLM, DNATokenizer),
    "dnaprom": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "distildna": (DistilBertConfig, DistilBertForMaskedLM, DNATokenizer),
    "distildnaprom": (DistilBertConfig, DistilBertForSequenceClassification, DNATokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "minidna": (MiniLMConfig, MiniLMForPreTraining, DNATokenizer),
    "minidnaprom": (BertConfig, BertForSequenceClassification, DNATokenizer),
}
MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}



def set_seed(args):
    """
    Set the seed for reproducibility
    
    :param args: This is the argument parser that we created in the previous section
    """
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



def main():
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--train_data_file", default=None, type=str, help="The input training data file (a text file).")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--teacher_model_type", type=str, default='dna', help="The teacher model architecture.")
    parser.add_argument("--student_model_type", type=str, default='distildna', help="The student model architecture.")
    parser.add_argument("--teacher_name_or_path", default=None, type=str, help="The pretrained teacher model.")
    parser.add_argument("--student_name_or_path", default=None, type=str, help="The model checkpoint for weights initialization or for evaluation")
    parser.add_argument("--student_config_name", default=None, type=str, help="Optional pretrained config name or path if not the same as student_name_or_path. If both are None, initialize a new config.")
    parser.add_argument("--tokenizer_name", default="dna6", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.")

    # OBJECTIVE
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_viz", action="store_true", help="Whether to plot attention maps on TATA promoters.")
    

    # DISTILLATION (general and task-specific)
    parser.add_argument("--alpha_ce", default=0.0, type=float, help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_mlm", default=1.0, type=float, help="Linear weight for the task loss (MLM or classification). Must be >=0.",)
    parser.add_argument("--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0.")
    parser.add_argument("--temperature", default=2.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")


    # GENERAL DISTILLATION
    parser.add_argument("--mlm", action="store_true", help="Train with masked-language modeling loss (general distillation), otherwise it will be task-specific distillation")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    

    # TASK-SPECIFIC (PROMOTER)
    parser.add_argument("--max_seq_length", type=int, default=384, help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--viz_steps", type=int, default=-1, help="Visualize attention every X updates steps.")


    # VALIDATION DURING TRAINING / EVALUATE 
    parser.add_argument("--eval_data_file", default=None, type=str, help="An optional input evaluation data file to evaluate the perplexity on (a text file).",)
    parser.add_argument("--metrics", action="store_true", help="Wether to calulate classification performance metrics during val/eval")    
    parser.add_argument("--do_val", action="store_true", help="If perform validation during training.",)
    parser.add_argument("--val_steps", type=int, default=500, help="Validate every X updates steps. Should be MULTIPLE of logging steps")


    # TRAINING DETAILS
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.98, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1 , type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_prop", default=0.0, type=float, help="Linear warmup proportioN.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_process", type=int, default=1, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--neptune", action="store_true", help="Neptune")
    parser.add_argument("--neptune_tags", type=list, default=["DistilBERT", "General", "trial"], help="Neptune")
    parser.add_argument("--neptune_description", type=str, default="TRIAL DistilBert general distillation", help="Neptune")
    parser.add_argument("--neptune_token", type=str, default=None, help="Neptune API token")
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    

    # OTHER 
    parser.add_argument("--cache_dir", default=None, type=str, help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",)
    parser.add_argument("--block_size", default=-1, type=int, help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).",)
    
    
    args = parser.parse_args()


    # INCOMPATIBILITIES -----------------------------------------------------------------------------------------------------

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.student_name_or_path is None and not args.do_train:
        raise ValueError(
            "If you're not going to train, please provide an already trained student model. Either supply a file to --student_name_or_path "
            "or insert the --do_train argument."
        )
    if args.teacher_name_or_path is None and args.do_train:
        raise ValueError(
            "Cannot do distillation without a teacher model. Either supply a file to --teacher_name_or_path "
            "or remove the --do_train argument."
        )

    if "prom" not in args.student_model_type and args.do_viz:
        raise ValueError(
            "Can only visualize attention maps of models trained for promoter identification. Either remove --do_viz"
            "or provide a suitable model (check --student_model_type)."
        )

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.student_name_or_path = sorted_checkpoints[-1]

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
    
    # MiniLM requires teacher model to be cast to minilm model type (same as bert but with desired outputs)
    if args.student_model_type == "minidna":
        args.teacher_model_type = "minidna"


    # SETUP -----------------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab


    # LOAD AND INITIALIZE MODELS -----------------------------------------------------------------------------------------------------
    
    # Teacher model
    if args.teacher_name_or_path:
        teacher_config_class, teacher_model_class, _ = MODEL_CLASSES[args.teacher_model_type]
        teacher_config = teacher_config_class.from_pretrained(args.teacher_name_or_path)
        teacher_config.output_hidden_states = True
        teacher = teacher_model_class.from_pretrained(args.teacher_name_or_path, config=teacher_config)
        teacher.to(args.device) 
    else:
        teacher = None


    # Student model
    student_config_class, student_model_class, tokenizer_class = MODEL_CLASSES[args.student_model_type]
    student_config = student_config_class.from_pretrained(args.student_config_name if args.student_config_name else args.student_name_or_path) # either you give the config or the semi trained model (if you want to continue training)
    if args.mlm:
        student_config.output_hidden_states = True
        student_config.output_attentions = False
    else:
        student_config.output_hidden_states = False
        student_config.output_attentions = True
 
    if args.student_name_or_path:
        student = student_model_class.from_pretrained(
            args.student_name_or_path,
            from_tf=bool(".ckpt" in args.student_name_or_path),
            config=student_config,
            cache_dir=args.cache_dir,
        )
        logger.info("Student model load")
    else:
        logger.info("Training new model from scratch")
        student = student_model_class(config=student_config)
        # Initalize student with teacher's weights (only DistilBERT)
        if args.student_model_type == 'distildna':
            init_student(teacher, student) 
  
    student.to(args.device)   


    # Tokenizer
    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)
        


    # TRAINING -----------------------------------------------------------------------------------------------------
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        # sanity checks
        assert student.config.vocab_size == teacher.config.vocab_size
        assert student.config.max_position_embeddings == teacher.config.max_position_embeddings
        
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.mlm:
            # General distillation
            train_dataset = load_and_cache_examples_pre(args, tokenizer, evaluate=False)
        else:
            # Task-specific distillation for DistilBERT (promoter)
            train_dataset = load_and_cache_examples_promoter(args, "dnaprom", tokenizer, evaluate=False)

        # If validation during training
        if args.do_val:
            if args.mlm:
                # General distillation
                val_dataset = load_and_cache_examples_pre(args, tokenizer, evaluate=True)
            else:
                # Task-specific distillation for DistilBERT (promoter)
                val_dataset = load_and_cache_examples_promoter(args, "dnaprom", tokenizer, evaluate=False, val=True)
        else:
            val_dataset = None
        
        if args.local_rank == 0:
            torch.distributed.barrier()

        # DISTILLER
        distiller = Distiller(params=args, train_dataset=train_dataset, student=student, teacher=teacher, tokenizer=tokenizer, val_dataset=val_dataset)
        distiller.train()
        logger.info("We're done!")


    # EVALUATION -----------------------------------------------------------------------------------------------------
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.mlm:
            # General distillation
            eval_dataset = load_and_cache_examples_pre(args, tokenizer, evaluate=True)
        else:
            # Task-specific distillation for DistilBERT (promoter)
            eval_dataset = load_and_cache_examples_promoter(args, "dnaprom", tokenizer, evaluate=True)

        if not args.do_train:
            distiller = Distiller(params=args, train_dataset=None, student=student, teacher=teacher, tokenizer=tokenizer)
        distiller.evaluate(eval_dataset)
    

    # ATTENTION VISUALIZATION -----------------------------------------------------------------------------------------------------
    if args.do_viz:
        distiller = Distiller(params=args, train_dataset=None, student=student, teacher=teacher, tokenizer=tokenizer)
        distiller.visualize_prom("viz.png")

if __name__ == "__main__":
    main()
