# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
Preprocessing script before training DistilBERT.
Specific to BERT -> DistilBERT.
"""


def init_student(teacher, student):
    
    prefix = "bert"
    
    s_params = dict(student.named_parameters())
    t_params = dict(teacher.named_parameters())
    
    for w in ["word_embeddings", "position_embeddings"]:
        s_params[f"distilbert.embeddings.{w}.weight"].data.copy_(t_params[f"{prefix}.embeddings.{w}.weight"].data)
    for w in ["weight", "bias"]:
        s_params[f"distilbert.embeddings.LayerNorm.{w}"].data.copy_(t_params[f"{prefix}.embeddings.LayerNorm.{w}"].data)

    std_idx = 0
    for teacher_idx in [0, 2, 4, 7, 9, 11]:
        for w in ["weight", "bias"]:
            s_params[f"distilbert.transformer.layer.{std_idx}.attention.q_lin.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.query.{w}"
            ].data)
            s_params[f"distilbert.transformer.layer.{std_idx}.attention.k_lin.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.key.{w}"
            ].data)
            s_params[f"distilbert.transformer.layer.{std_idx}.attention.v_lin.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.value.{w}"
            ].data)

            s_params[f"distilbert.transformer.layer.{std_idx}.attention.out_lin.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"
            ].data)
            s_params[f"distilbert.transformer.layer.{std_idx}.sa_layer_norm.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"
            ].data)

            s_params[f"distilbert.transformer.layer.{std_idx}.ffn.lin1.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"
            ].data)
            s_params[f"distilbert.transformer.layer.{std_idx}.ffn.lin2.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.output.dense.{w}"
            ].data)
            s_params[f"distilbert.transformer.layer.{std_idx}.output_layer_norm.{w}"].data.copy_(t_params[
                f"{prefix}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"
            ].data)
        std_idx += 1

