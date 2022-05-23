# Knowledge Distillaton of DNABERT for Prediction of Genomic Elements

This repository includes the implementation of "Knowledge Distillaton of DNABERT for Prediction of Genomic Elements". It includes source codes for data acquisition, distillation and fine-tuning of student models and usage examples. Pre-trained and fine-tuned models will be available soon.

All the models were build upon the framework provided by HuggingFace. Parts of the code provided by the authors of [DNABERT](https://github.com/jerryji1993/DNABERT), [DistilBERT](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) and an unoﬀicial reimplementation of [MiniLM](https://github.com/jongwooko/Pytorch-MiniLM) served as a base for this work and were extended to fit the specific needs.

## Usage examples

Below you can find examples on how to perform basic distillations, promoter fine-tuning and evaluation. 
Check allowed arguments to find more advanced options, e.g. `python run_distil.py -h`.

### Pre-train DistilBERT student model

```bash
python 'run_distil.py' \
    --train_data_file sample_6_3k.txt \
    --output_dir models \
    --student_model_type distildna \
    --student_config_name src/transformers/dnabert-config/distilbert-config-6 \
    --teacher_name_or_path models/dnabert/6mer_pretrained \
    --mlm \
    --do_train \
    --learning_rate 0.0004 \
    --logging_steps 500 \
    --save_steps 8000 \
    --num_train_epochs 2
```

### Pre-train MiniLM student model

```bash

```

### Fine-tune for promoter identification
```bash

```

### DistilBERT additional distillation for promoter identification



### Classification metrics evaluation on test set
```bash

```

### Attention landscapes for TATA-promoters
```bash

```

