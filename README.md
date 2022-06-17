# Knowledge Distillaton of DNABERT for Prediction of Genomic Elements

This repository includes the implementation of "Knowledge Distillaton of DNABERT for Prediction of Genomic Elements". It includes source codes for data acquisition, distillation and fine-tuning of student models and usage examples. Pre-trained and fine-tuned models will be available soon.

All the models were build upon the framework provided by HuggingFace. Parts of the code provided by the authors of [DNABERT](https://github.com/jerryji1993/DNABERT), [DistilBERT](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) and an unoﬀicial reimplementation of [MiniLM](https://github.com/jongwooko/Pytorch-MiniLM) served as a base for this work and were extended to fit the specific needs.

For more information on the methodology and results, please check the full document [link to be added].

## Pre-trained models

|   **Model**  | **# Layers** | **# Hidden units** |                                            **Link**                                            |
|:------------:|:------------:|:------------------:|:----------------------------------------------------------------------------------------------:|
|    DNABERT   |      12      |         768        | *Made available by the authors. Official github [here](https://github.com/jerryji1993/DNABERT) |
|  DistilBERT  |       6      |         768        |            [dnabert-distilbert](https://huggingface.co/Peltarion/dnabert-distilbert)           |
|    MiniLM    |       6      |         768        |                [dnabert-minilm](https://huggingface.co/Peltarion/dnabert-minilm)               |
| MiniLM small |       6      |         384        |          [dnabert-minilm-small](https://huggingface.co/Peltarion/dnabert-minilm-small)         |
|  MiniLM mini |       3      |         384        |           [dnabert-minilm-mini](https://huggingface.co/Peltarion/dnabert-minilm-mini)          |


## Usage examples

Below you can find examples on how to perform basic distillations, promoter fine-tuning and evaluation. 
Check allowed arguments to find more advanced options, e.g. `python run_distil.py -h`.

### Pre-train DistilBERT student model

```bash
python run_distil.py \
    --train_data_file data/pretrain/sample_6_3k.txt \
    --output_dir models \
    --student_model_type distildna \
    --student_config_name src/transformers/dnabert-config/distilbert-config-6 \
    --teacher_name_or_path Path_to_pretrained_DNABERT \
    --mlm \
    --do_train \
    --alpha_ce 2 \
    --alpha_mlm 7 \
    --alpha_cos 1 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 0.0004 \
    --logging_steps 500 \
    --save_steps 8000 \
    --num_train_epochs 2
```

### Pre-train MiniLM student model

```bash
python run_distil.py \
    --train_data_file data/pretrain/sample_6_3k.txt \
    --output_dir models \
    --student_model_type minidna \
    --student_config_name src/transformers/dnabert-config/minilm-config-6 \
    --teacher_name_or_path Path_to_pretrained_DNABERT \
    --mlm \
    --do_train \
    --per_gpu_train_batch_size 32 \
    --learning_rate 0.0004 \
    --logging_steps 500 \
    --save_steps 8000 \
    --num_train_epochs 2
```

### DistilBERT additional distillation for promoter identification

_Before running the script, process promoter dataset to obtain the training data using `porcess_finetune_data.py`_

```bash
python run_distil.py \
    --train_data_file data/promoters/6mer \
    --output_dir models \
    --student_model_type distildnaprom \
    --student_name_or_path Peltarion/dnabert-distilbert \
    --teacher_model_type dnaprom \
    --teacher_name_or_path Path_to_finetuned_DNABERT \
    --do_train \
    --alpha_ce 1 \
    --alpha_mlm 1 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 0.00005 \
    --logging_steps 500 \
    --save_steps 1000 \
    --num_train_epochs 3 \
    --do_val \
    --eval_data_file data/promoters/6kmer
```
    
### Fine-tune for promoter identification

_Before running the script, process promoter dataset to obtain the training data using `porcess_finetune_data.py`_

```bash
python run_finetune.py \
    --data_dir data/promoters/6mer \
    --output_dir models \
    --model_type distildnaprom \
    --model_name_or_path Peltarion/dnabert-distilbert \
    --do_train \
    --per_gpu_train_batch_size 32 \
    --learning_rate 0.00005 \
    --logging_steps 100 \
    --save_steps 1000 \
    --num_train_epochs 3 \
    --evaluate_during_training 
```

### Promoter prediction on test set

_Example with fine-tuned DNABERT_

```bash
python run_finetune.py \
    --data_dir data/promoters/6mer \
    --output_dir models \
    --do_predict \
    --model_type dnaprom \
    --model_name_or_path Path_to_finetuned_DNABERT \
    --per_gpu_eval_batch_size 32  
```

### Classification metrics evaluation on test set

_Example with fine-tuned MiniLM_

```bash
python run_finetune.py \
    --data_dir data/promoters/6mer \
    --output_dir models \
    --do_eval \
    --model_type minidnaprom \
    --model_name_or_path Path_to_finetuned_MiniLM \
    --per_gpu_eval_batch_size 32  
```

### Attention landscapes for TATA-promoters

_Example with fine-tuned DistilBERT_

```bash
python run_finetune.py \
    --data_dir data/promoters/6mer \
    --output_dir models \
    --do_visualize \
    --model_type distildnaprom \
    --model_name_or_path Path_to_finetuned_DistilBERT \
    --per_gpu_eval_batch_size 32  
```

