# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
The distiller to distil the student.
"""
import math
import os
import time
from typing import Dict, List, Tuple
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from data_loaders import *

from transformers import get_linear_schedule_with_warmup
from src.transformers import glue_compute_metrics as compute_metrics
from utils import logger
import neptune.new as neptune


class Distiller:
    def __init__(self, params: dict, train_dataset, student, teacher, tokenizer, val_dataset=None):

        logger.info("Initializing Distiller")
        self.params = params
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size
        self.tokenizer = tokenizer

        self.mlm = params.mlm

        self.tata_dataset=None
        
        if params.do_train:
            self.train_batch_size = params.per_gpu_train_batch_size * max(1, params.n_gpu)

            if params.local_rank == -1:
                sampler = RandomSampler(train_dataset)
            else:
                sampler = DistributedSampler(train_dataset)

            self.num_examples = len(train_dataset)
            if self.mlm:
                self.dataloader = DataLoader(dataset=train_dataset, sampler=sampler, batch_size=self.train_batch_size, collate_fn=self.collate)
            else:
                self.dataloader = DataLoader(dataset=train_dataset, sampler=sampler, batch_size=self.train_batch_size)
            
            if val_dataset is not None:
                eval_sampler = SequentialSampler(val_dataset)
                if self.mlm:
                    self.eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=self.params.per_gpu_eval_batch_size, collate_fn=self.collate)
                else:
                    self.eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=self.params.per_gpu_eval_batch_size)

            if params.student_model_type != 'minidna':
                self.temperature = params.temperature
                assert self.temperature > 0.0
                self.alpha_ce = params.alpha_ce
                self.alpha_mlm = params.alpha_mlm
                self.alpha_cos = params.alpha_cos
                self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
                self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                self.last_loss_ce = 0
                if self.alpha_cos > 0.0:
                    self.last_loss_cos = 0
                    self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
            else: #minidna
                self.last_loss_at = 0
                self.last_loss_vr = 0
                self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
                self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            self.epoch = 0
            self.n_iter = 0
            self.n_total_iter = 0
            self.n_sequences_epoch = 0
            self.total_loss_epoch = 0
            self.last_loss = 0
            self.last_loss_s = 0

            self.last_epoch = 0

            logger.info("--- Initializing model optimizer")
            assert params.gradient_accumulation_steps >= 1
            
            
            self.num_steps_epoch = len(self.dataloader)
            self.num_train_optimization_steps = self.num_steps_epoch // params.gradient_accumulation_steps * params.num_train_epochs

                
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": params.weight_decay,
                },
                {
                    "params": [
                        p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            logger.info(
                "------ Number of trainable parameters (student): %i"
                % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
            )
            logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
            self.optimizer = AdamW(
                optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(params.beta1,params.beta2)
            )

            if params.warmup_steps == 0:
                self.params.warmup_steps = math.ceil(self.num_train_optimization_steps * params.warmup_prop)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.params.warmup_steps, num_training_steps=self.num_train_optimization_steps
            )

            # Check if saved optimizer or scheduler states exist
            if (
                params.student_name_or_path
                and os.path.isfile(os.path.join(params.student_name_or_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(params.student_name_or_path, "scheduler.pt"))
            ):
                # Load in optimizer and scheduler states
                self.optimizer.load_state_dict(torch.load(os.path.join(params.student_name_or_path, "optimizer.pt")))
                self.scheduler.load_state_dict(torch.load(os.path.join(params.student_name_or_path, "scheduler.pt")))

            # Mixed precision training
            if self.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
                self.student, self.optimizer = amp.initialize(
                    self.student, self.optimizer, opt_level=self.params.fp16_opt_level
                )
                self.teacher = self.teacher.half()

    
        # multi-gpu training
        if params.n_gpu > 1:
            self.student = torch.nn.DataParallel(self.student)
            if self.teacher is not None:
                self.teacher = torch.nn.DataParallel(self.teacher)

        # Distributed training
        if params.local_rank != -1:
            self.student = torch.nn.parallel.DistributedDataParallel(
                self.student, device_ids=[params.local_rank], output_device=params.local_rank, find_unused_parameters=True
            )
            if self.teacher is not None:
                self.teacher = torch.nn.parallel.DistributedDataParallel(
                    self.teacher, device_ids=[params.local_rank], output_device=params.local_rank, find_unused_parameters=True
                )

        self.neptune = params.neptune

    

    def collate(self, examples: List[torch.Tensor]):
        """
        Takes a list of tensors and returns a single tensor
        
        :param examples: List[torch.Tensor]
        :return: A list of tensors.
        """
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. 
        
        :param inputs: the input sequence
        :return: The input, the padding mask and the labels
        """
        MASK_LIST = {
            "3": [-1, 1],
            "4": [-1, 1, 2],
            "5": [-2, -1, 1, 2],
            "6": [-2, -1, 1, 2, 3]
        }
        
        mask_list = MASK_LIST[self.tokenizer.kmer]

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.params.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0) # 0 prob of sampling special tokens
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        padding_mask = ~padding_mask # For loss calculation, need a tensor False if padding

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Because of the sliding window tokenization strategy we need to mask also sorrounding tokens, otherwise the solution of the problem would be trivial
        masks = deepcopy(masked_indices)
        for i, masked_index in enumerate(masks):
            end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
            mask_centers = set(torch.where(masked_index==1)[0].tolist())
            new_centers = deepcopy(mask_centers)
            for center in mask_centers:
                for mask_number in mask_list:
                    current_index = center + mask_number
                    if current_index <= end and current_index >= 1:
                        new_centers.add(current_index)
            new_centers = list(new_centers)
            masked_indices[i][new_centers] = True
        

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, padding_mask, labels 



    def train(self):
        """
        The real training loop.
        """
        logger.info("Starting training")
        
        if self.neptune:
            logger.info("--- Initializing Neptune")
            self.run = neptune.init(
                project=self.params.neptune_project,
                api_token=self.params.neptune_token,
                tags=self.params.neptune_tags,
                description=self.params.neptune_description,
            )
            self.run["model/parameters"] = vars(self.params)
            self.run["model/student_config"] = vars(self.student_config)
        
        self.last_epoch = time.time()
        self.student.train()
        self.teacher.eval()
        
        # Training details
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples)
        logger.info("  Num Epochs = %d", self.params.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.params.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.train_batch_size
            * self.params.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.params.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.params.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.num_train_optimization_steps)
        
        
        steps_trained_in_current_epoch = 0
        
        # Check if continuing training from a checkpoint
        if self.params.student_name_or_path and os.path.exists(self.params.student_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.params.student_name_or_path.split("-")[-1].split("/")[0]
                self.n_total_iter = int(checkpoint_suffix)
                self.epoch = self.n_total_iter // self.num_steps_epoch
                steps_trained_in_current_epoch = self.n_total_iter % self.num_steps_epoch

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", self.epoch)
                logger.info("  Continuing training from global step %d", self.n_total_iter)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")
        
        # Training loop
        train_iterator = trange(
            self.epoch, int(self.params.num_train_epochs), desc="Epoch", disable=self.params.local_rank not in [-1, 0]
        )

        for _ in train_iterator:
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.num_train_epochs-1}")

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                if self.mlm:
                    token_ids, attn_mask, labels = self.mask_tokens(inputs=batch)
                else:
                    # Task-specific distillation for DistilBERT (promoter)
                    token_ids, attn_mask, labels = batch[0], batch[1], batch[3]
                
                token_ids = token_ids.to(self.params.device)
                attn_mask = attn_mask.to(self.params.device)
                labels = labels.to(self.params.device)
                
                if self.params.student_model_type != 'minidna':
                    self.step(input_ids=token_ids, attention_mask=attn_mask, labels=labels)
                else: #minidna
                    self.step_minilm(input_ids=token_ids, attention_mask=attn_mask, labels=labels)

                iter_bar.update()

            iter_bar.close()
            train_iterator.update()

            logger.info(f"--- Ending epoch {self.epoch}/{self.params.num_train_epochs-1}")
            self.end_epoch()
            self.last_epoch = time.time()
        
        train_iterator.close()

        # Save the final model
        logger.info("Save very last checkpoint")
        self.save_checkpoint(checkpoint_prefix="final")
        logger.info("Training is finished")


    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, val=False):
        """
        One optimization step DisitlBERT: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        labels: `torch.tensor(bs, seq_length)` - The corresponding labels (mlm labels for MLM or classification).
        """
        losses_dict = {}
        if self.mlm:
            student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask.long())  # (bs, seq_length, voc_size)
            s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
            t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        else:
            student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask.long())  # (bs, seq_length, voc_size)
            s_logits, s_loss = student_outputs["logits"], student_outputs["loss"]
            t_logits = teacher_outputs["logits"] # Don't need to evaluate the teacher's performance

        assert s_logits.size() == t_logits.size()

        # Calculate distillation loss
        if self.mlm:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
            # Take only the logits of the elements that are not padding
            s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
            t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            assert t_logits_slct.size() == s_logits_slct.size()
        else:
            s_logits_slct = s_logits
            t_logits_slct = t_logits

        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce
        losses_dict["ce_loss"] = loss_ce.item()

        if self.mlm:
            s_loss = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), labels.view(-1)) # Cross entropy amb les labels, hem especificat ignorar els -100 (label dels tokens q no hem canviat)
            loss += self.alpha_mlm * s_loss
        else:
            loss += self.alpha_mlm * s_loss
        losses_dict["task_loss"] = s_loss.item()

        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos
            losses_dict["cos_loss"] = loss_cos.item()

        losses_dict["total_loss"] = loss.item()
            
        if val:
            return losses_dict, s_logits

        else:
            self.total_loss_epoch += loss.item()
            self.last_loss = loss.item()
            self.last_loss_ce = loss_ce.item()
            self.last_loss_s = s_loss.item()
            if self.alpha_cos > 0.0:
                self.last_loss_cos = loss_cos.item()

            self.optimize(loss)

            self.n_sequences_epoch += input_ids.size(0)


    def step_minilm(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, val=False):
        """
        One optimization step for MiniLM: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        labels: `torch.tensor(bs, seq_length)` - The corresponding labels (mlm labels for MLM or classification).
        """
        losses_dict = {}
        if self.mlm:
            # Get student attention matrices
            prediction_scores_s, _, student_query_layers, student_key_layers, student_value_layers = self.student(input_ids=input_ids, attention_mask=attention_mask, is_student=True)
                
            student_attention_dist = torch.matmul(student_query_layers[-1], student_key_layers[-1].transpose(-1,-2))
            student_value_relation = torch.matmul(student_value_layers[-1], student_value_layers[-1].transpose(-1,-2))
            student_attention_head_size = int(self.student.config.hidden_size / self.student.config.num_attention_heads)

            student_attention_dist = student_attention_dist / math.sqrt(student_attention_head_size)
            student_attention_dist = F.log_softmax(student_attention_dist, dim=-1)

            student_value_relation = student_value_relation / math.sqrt(student_attention_head_size)
            student_value_relation = F.log_softmax(student_value_relation, dim=-1)
            
            with torch.no_grad():
                # Get teacher attention matrices
                _, _, teacher_query_layers, teacher_key_layers, teacher_value_layers = self.teacher(input_ids=input_ids, attention_mask=attention_mask, is_student=False)
                
                teacher_attention_dist = torch.matmul(teacher_query_layers[-1], teacher_key_layers[-1].transpose(-1,-2))
                teacher_value_relation = torch.matmul(teacher_value_layers[-1], teacher_value_layers[-1].transpose(-1,-2))
                teacher_attention_head_size = int(self.teacher.config.hidden_size / self.teacher.config.num_attention_heads)
                
                teacher_attention_dist = teacher_attention_dist / math.sqrt(teacher_attention_head_size)
                teacher_attention_dist = F.log_softmax(teacher_attention_dist, dim=-1)
                
                teacher_value_relation = teacher_value_relation / math.sqrt(teacher_attention_head_size)
                teacher_value_relation = F.log_softmax(teacher_value_relation, dim=-1)

        else:
            print("Not prepared for any other type of distillation. Set --mlm to True")


        # Calculate distillation loss
        scaler = self.teacher.config.num_attention_heads*teacher_value_relation.size(dim=-1)
        at_loss = self.ce_loss_fct(teacher_attention_dist, student_attention_dist) / scaler
        vr_loss = self.ce_loss_fct(teacher_value_relation, student_value_relation) / scaler
        loss = at_loss + vr_loss

        s_loss = self.lm_loss_fct(prediction_scores_s.view(-1, prediction_scores_s.size(-1)), labels.view(-1)) 
            
        if val:
            losses_dict["at_loss"] = at_loss.item()
            losses_dict["vr_loss"] = vr_loss.item()
            losses_dict["task_loss"] = s_loss.item()
            losses_dict["total_loss"] = loss.item()
            return losses_dict, _

        else:
            self.total_loss_epoch += loss.item()
            self.last_loss = loss.item()
            self.last_loss_at = at_loss.item()
            self.last_loss_vr = vr_loss.item()
            self.last_loss_s = s_loss.item()

            self.optimize(loss)

            self.n_sequences_epoch += input_ids.size(0)


    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.params.n_gpu > 1:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()


        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to neptune and save checkpoint.
        """

        if self.n_total_iter % self.params.logging_steps == 0:
            val_losses = None
            if self.params.do_val and self.n_total_iter % self.params.val_steps == 0:
                val_losses = self.val_loss()
            self.log_neptune(val=val_losses)
        if self.n_total_iter % self.params.save_steps == 0:
            self.save_checkpoint()

        if self.params.viz_steps != -1 and self.n_total_iter % self.params.viz_steps == 0:
            self.visualize_prom()

        self.n_iter += 1 # Step
        self.n_total_iter += 1 # Global step

    def log_neptune(self, val=None):
        """
        Log into neptune.
        """
        if not self.neptune:
            return

        self.run["train/loss"].log(self.last_loss, step=self.n_total_iter)
        self.run["train/loss_mlm"].log(self.last_loss_s, step=self.n_total_iter) # if mlm, else it's just the task loss
        self.run["train/lr"].log(self.scheduler.get_lr()[0], step=self.n_total_iter)

        if self.params.student_model_type != 'minidna':
            #Specific for distilBERT
            self.run["train/loss_ce"].log(self.last_loss_ce, step=self.n_total_iter)
            if self.alpha_cos > 0.0:
                self.run["train/loss_cos"].log(self.last_loss_cos, step=self.n_total_iter)
        else:
            #Specific for miniLM
            self.run["train/loss_at"].log(self.last_loss_at, step=self.n_total_iter)
            self.run["train/loss_vr"].log(self.last_loss_vr, step=self.n_total_iter)
        
        if val is not None:
            for key in val[0].keys():
                self.run["val/"+key].log(val[0][key], step=self.n_total_iter)
            if not self.mlm:
                for key in val[-1].keys():
                    self.run["val/"+key].log(val[-1][key], step=self.n_total_iter)


    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some logging and checkpoint saving.
        """
        #logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.neptune:
            self.run["train/epoch_minutes"].log((time.time() - self.last_epoch)/60, step=self.n_total_iter)

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_prefix: str = "checkpoint"):
        """
        Save the current state. Only by the master process.
        """
        if checkpoint_prefix == "checkpoint":
            output_dir = os.path.join(self.params.output_dir, "{}-{}".format(checkpoint_prefix, self.n_total_iter))
        else:
            output_dir = self.params.output_dir
            
        # Save model checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = (self.student.module if hasattr(self.student, "module") else self.student)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(self.params, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        if checkpoint_prefix == "checkpoint":
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            
    
    def val_loss(self):

        # multi-gpu evaluate
        if self.params.n_gpu > 1 and not isinstance(self.student, torch.nn.DataParallel):
            self.student = torch.nn.DataParallel(self.student)
            self.teacher = torch.nn.DataParallel(self.teacher)
        self.student.eval()
        self.teacher.eval()

        n_batches = len(self.eval_dataloader)

        logger.info("***** Running validation *****")
        logger.info("  Num batches = %d", n_batches)
        logger.info("  Batch size = %d", self.params.per_gpu_eval_batch_size)


        losses_dict = {}
        metrics_dict = {}
        preds = None
        probs = None
        out_label_ids = None
        softmax = torch.nn.Softmax(dim=1)
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            if self.mlm:
                token_ids, attn_mask, labels = self.mask_tokens(inputs=batch)
            else:
                # For task-specific distillation
                token_ids, attn_mask, labels = batch[0], batch[1], batch[3]
            token_ids = token_ids.to(self.params.device)
            attn_mask = attn_mask.to(self.params.device)
            labels = labels.to(self.params.device)

            if self.params.student_model_type != 'minidna':
                temp_val_losses, logits_s = self.step(input_ids=token_ids, attention_mask=attn_mask, labels=labels, val=True)
            else: #minidna
                temp_val_losses, logits_s = self.step_minilm(input_ids=token_ids, attention_mask=attn_mask, labels=labels, val=True)


            if not losses_dict:
                for key in temp_val_losses.keys():
                    losses_dict[key] = temp_val_losses[key]/n_batches
            else:
                for key in temp_val_losses.keys():
                    losses_dict[key] += temp_val_losses[key]/n_batches
            

            # To calculate classification performance metrics
            if not self.mlm:
                if preds is None:
                    preds = logits_s.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits_s.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        
        if not self.mlm:
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
            preds = np.argmax(preds, axis=1)
            metrics_dict = compute_metrics("dnaprom", preds, out_label_ids, probs)

        self.student.train()
        
        return losses_dict, metrics_dict


    def visualize_prom(self,save_path=None):
        if self.tata_dataset is None:
            self.tata_dataset = load_and_cache_examples_promoter(self.params, "dnaprom", self.tokenizer, viz=True)
        kmer = int(self.params.tokenizer_name[-1])
        attention_scores, _ = visualize(self.params, self.student, self.tokenizer, prefix="", kmer=kmer, pred_dataset=self.tata_dataset)
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
        if self.neptune:
            self.run["train/attention"].log(fig, step=self.n_total_iter)
        if isinstance(save_path,str):
            plt.savefig(save_path)


    def evaluate(self, eval_dataset, prefix="") -> Dict:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = self.params.output_dir
        softmax = torch.nn.Softmax(dim=1)

        if self.params.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        self.params.eval_batch_size = self.params.per_gpu_eval_batch_size * max(1, self.params.n_gpu)
        # Note that DistributedSampler samples randomly

        eval_sampler = SequentialSampler(eval_dataset)
        if self.mlm:
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.params.eval_batch_size, collate_fn=self.collate)
        else:
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.params.eval_batch_size)

        # multi-gpu evaluate
        if self.params.n_gpu > 1 and not isinstance(self.student, torch.nn.DataParallel):
            self.student = torch.nn.DataParallel(self.student)
            self.student.eval()
            if self.teacher is not None:
                self.teacher = torch.nn.DataParallel(self.teacher)
                self.teacher.eval()

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.params.eval_batch_size)
        eval_loss_s = 0.0
        eval_loss_t = 0.0
        nb_eval_steps = 0
        result = {}
        preds = None
        probs = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if self.mlm:
                token_ids, attn_mask, labels = self.mask_tokens(inputs=batch)
            else:
                # For sentence classification task
                token_ids, attn_mask, labels = batch[0], batch[1], batch[3]
                if self.params.student_model_type != "distildna" and self.params.student_model_type != "distildnaprom":
                    token_type_ids = batch[2]
                    token_type_ids = token_type_ids.to(self.params.device)
            token_ids = token_ids.to(self.params.device)
            attn_mask = attn_mask.to(self.params.device)
            labels = labels.to(self.params.device)

            with torch.no_grad():
                if self.params.student_model_type != "distildna" and self.params.student_model_type != "distildnaprom" :
                    outputs_s = self.student(token_ids, attention_mask=attn_mask, labels=labels, token_type_ids=token_type_ids) 
                else:
                    outputs_s = self.student(token_ids, attention_mask=attn_mask, labels=labels) 
                lm_loss_s, logits_s = outputs_s["loss"], outputs_s["logits"]
                eval_loss_s += lm_loss_s.mean().item()
                if self.teacher is not None:
                    outputs_t = self.teacher(token_ids, attention_mask=attn_mask, labels=labels) 
                    lm_loss_t = outputs_t["loss"]
                    eval_loss_t += lm_loss_t.mean().item()
            nb_eval_steps += 1
            if not self.mlm:
                if preds is None:
                    preds = logits_s.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits_s.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

            
        eval_loss_s = eval_loss_s / nb_eval_steps
        perplexity_s = torch.exp(torch.tensor(eval_loss_s))
        result = {"perplexity_student": perplexity_s}
        if not self.mlm:
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
            preds = np.argmax(preds, axis=1)
            metrics_dict = compute_metrics("dnaprom", preds, out_label_ids, probs)
            result.update(metrics_dict)

        if self.neptune:
            for key in result.keys():
                self.run["test/"+key] = result[key]
            
        if self.teacher is not None:
            eval_loss_t = eval_loss_t / nb_eval_steps
            perplexity_t = torch.exp(torch.tensor(eval_loss_t))
            result["perplexity_teacher"] = perplexity_t
            if self.neptune:
                self.run["test/perplexity_teacher"] = result["perplexity_teacher"]

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write(str(float(result[key])) + "\n")

            
        

