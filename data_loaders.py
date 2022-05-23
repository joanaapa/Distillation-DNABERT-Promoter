import os
import torch
import pickle
from multiprocessing import Pool
import numpy as np

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import PreTrainedTokenizer
from utils import logger
from tqdm import tqdm

from src.transformers import glue_convert_examples_to_features as convert_examples_to_features
from src.transformers import glue_output_modes as output_modes
from src.transformers import glue_processors as processors

# Pretrain/General distillation
class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.student_model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)["input_ids"]
    return examples

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        #assert os.path.isfile(file_path) TODO: uncomment this
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.student_model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            
            if args.n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
            else:
                n_proc = args.n_process
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(lines)/n_proc)
                for i in range(1, n_proc+1):
                    if i != n_proc:
                        indexes.append(len_slice*(i))
                    else:
                        indexes.append(len(lines))
                results = []
                for i in range(n_proc):
                    results.append(p.apply_async(convert_line_to_example,[tokenizer, lines[indexes[i]:indexes[i+1]], block_size,]))
                    print(str(i) + " start")
                p.close() 
                p.join()

                self.examples = []
                for result in results:
                    ids = result.get()
                    self.examples.extend(ids)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def load_and_cache_examples_pre(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)




# Promoter dataset
def load_and_cache_examples_promoter(args, task, tokenizer, evaluate=False, val=False, viz=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # For compatibility between two scripts
    try:
        print(args.data_dir)
    except AttributeError:
        if evaluate or val:
            args.data_dir = args.eval_data_file
        else:
            args.data_dir = args.train_data_file

    try:
        print(args.model_name_or_path)
    except AttributeError:
        args.model_name_or_path = args.student_name_or_path

    try:
        print(args.do_predict)
    except AttributeError:
        args.do_predict = False



    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "test" if evaluate else "dev" if val else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if args.do_predict:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "test" if evaluate else "dev" if val else "train", str(args.max_seq_length), str(task),
            ),
        )
    
    if viz:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "tata", str(300), str(task),
            ),
        )
    
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        
        if viz:
            examples = (processor.get_tata_examples(args.data_dir))
        else:
            examples = (
                processor.get_test_examples(args.data_dir)
                if evaluate
                else processor.get_dev_examples(args.data_dir)
                if val
                else processor.get_train_examples(args.data_dir)
            )

        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length if not viz else 300
        pad_on_left = False
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 0

        if args.n_process == 1:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id,
            )

        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc / 4), 1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples) / n_proc)
            for i in range(1, n_proc + 1):
                if i != n_proc:
                    indexes.append(len_slice * (i))
                else:
                    indexes.append(len(examples))

            results = []

            for i in range(n_proc):
                results.append(
                    p.apply_async(
                        convert_examples_to_features,
                        args=(
                            examples[indexes[i] : indexes[i + 1]],
                            tokenizer,
                            max_length,
                            None,
                            label_list,
                            output_mode,
                            pad_on_left,
                            pad_token,
                            pad_token_segment_id,
                            True,
                        ),
                    )
                )
                print(str(i + 1) + " processor started !")

            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def visualize(args, model, tokenizer, kmer, prefix="", pred_dataset=None):

    # For compatibility between two scripts
    try:
        print(args.per_gpu_eval_batch_size)
    except AttributeError:
        args.per_gpu_eval_batch_size = args.per_gpu_pred_batch_size


    softmax = torch.nn.Softmax(dim=1)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running visualization {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    pred_loss = 0.0
    nb_pred_steps = 0
    batch_size = args.eval_batch_size

    preds = np.zeros([len(pred_dataset), 2])
    max_seq_length = 300

    attention_scores = np.zeros([len(pred_dataset), 12, max_seq_length, max_seq_length])

    for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            attention = outputs[-1][-1]
            _, logits = outputs[:2]

            preds[index * batch_size : index * batch_size + len(batch[0]), :] = logits.detach().cpu().numpy()
            attention_scores[index * batch_size : index * batch_size + len(batch[0]), :, :, :] = attention.cpu().numpy()

    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

    for index, attention_score in enumerate(attention_scores):
        attn_score = []
        for i in range(1, attention_score.shape[-1] - kmer + 2):
            attn_score.append(float(attention_score[:, 0, i].sum()))

        for i in range(len(attn_score) - 1):
            if attn_score[i + 1] == 0:
                attn_score[i] = 0
                break

        counts = np.zeros([len(attn_score) + kmer - 1])
        real_scores = np.zeros([len(attn_score) + kmer - 1])
        for i, score in enumerate(attn_score):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score
        real_scores = real_scores / counts
        real_scores = real_scores / np.linalg.norm(real_scores)

        scores[index] = real_scores

    return scores, probs