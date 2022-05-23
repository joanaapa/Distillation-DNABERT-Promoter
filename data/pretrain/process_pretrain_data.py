"""
Construct the fine-tuning dataset. Adapted from DNABERT
"""

import argparse
import random
from tkinter import N
from pyfaidx import Fasta
from tqdm import tqdm

import numpy as np


def cut_no_overlap(length, kmer=1, max_prob=0.5):
    print("---Getting the cut indices...")
    cuts = []
    while length:
        if length <= 509 + kmer:
            cuts.append(length)
            break
        else:
            if random.random() > max_prob:
                cut = max(int(random.random() * (509 + kmer)), 10)
            else:
                cut = 509 + kmer
            cuts.append(cut)
            length -= cut

    return cuts


def sampling(length, kmer=1, sampling_rate=1):
    times = int(length * sampling_rate / 256)
    starts = []
    ends = []
    for i in range(times):
        cut = max(int(random.random() * (509 + kmer)), 10)
        start = np.random.randint(length - kmer)
        starts.append(start)
        ends.append(start + cut)

    return starts, ends


def sampling_fix(length, kmer=1, sampling_rate=1, fix_length=10245):
    times = int(length * sampling_rate / fix_length)
    starts = []
    ends = []
    for i in range(times):
        cut = fix_length
        start = np.random.randint(length - 6 - fix_length)
        starts.append(start)
        ends.append(start + cut)

    return starts, ends


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer:
        sentence += original_string[i : i + kmer] + " "
        i += stride

    return sentence[:-1].strip('"')


def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string) - kmer):
        sequence.append(original_string[i : i + kmer])

    sequence.append(original_string[-kmer:])
    return sequence


def Process(args, sample=None):

    data = Fasta(args.file_path)

    if args.output_path is None:
        args.output_path = args.file_path

    if args.sampling_rate != 1.0:
        strategy = "_sam"
    else:
        strategy = "_cut"
    
    new_file_path = args.output_path + strategy + str(args.kmer) + ".txt"
    new_file = open(new_file_path, "w")


    seq_counts = {}
    n_total = 0
    for idx in tqdm(range(1,23), desc="chromosomes"):
        n_chrom = 0

        # Save chromosomes 7 and 8 for val/test
        if args.val:
            if idx not in [7,8]:
                print("Chromosome skipped ->", idx)
                continue 
        else:
            if not idx == 7 and not idx == 8:
                print("Chromosome skipped ->", idx)
                continue 
    
        line = data["chr"+str(idx)][:].seq.upper()
        line_length = len(line)

        if args.sampling_rate != 1.0:
            # Random sampling strategy
            starts, ends = sampling_fix(
                length=line_length, kmer=args.kmer, sampling_rate=args.sampling_rate, fix_length=args.length
            )
            for i in range(len(starts)):
                new_line = line[starts[i] : ends[i]]
                sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                new_file.write(sentence + "\n")

        else:
            # Cut without overlap strategy
            cuts = cut_no_overlap(length=line_length, kmer=args.kmer)
            start = 0
            print("---Extracting sequences...")
            for i,cut in enumerate(tqdm(cuts, desc="Seq")):
                new_line = line[start : start + cut]
                # Filter out the sequences that contain "N" (unknown bases in ref. genome)
                if "N" not in new_line:
                    if i % args.val_sampling ==0:
                        sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                        new_file.write(sentence + "\n")
                        n_chrom += 1
                start += cut

        seq_counts["chr"+str(idx)] = n_chrom
        n_total += n_chrom
    seq_counts["TOTAL"] = n_total

    # Save number of sampled sequences
    statistics_path = args.output_path + strategy + str(args.kmer) + "_statistics.txt"
    with open(statistics_path, "w") as f:
        print(seq_counts, file=f)

    print("NUMBER SEQUENCES EXTRACTED IS ", n_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_rate", default=1.0, type=float, help="We will sample sampling_rate*total_length*2/512 times",)
    parser.add_argument("--kmer", default=1, type=int, help="K-mer",)
    parser.add_argument("--length", default=10000, type=int, help="Length of the sampled sequence",)
    parser.add_argument("--file_path", default=None, type=str, help="The path of the file to be processed",)
    parser.add_argument("--output_path", default=None, type=str, help="The path of the processed data",)
    parser.add_argument("--val", action="store_true", help="To construct validation set",)
    parser.add_argument("--val_sampling", default=1, type=int, help="To get a subset of the data, for val",)

    args = parser.parse_args()

    # Args prepared to cut and no overlap
    args.file_path = "/mnt/storage/data/joana_pales/msc-joana/data/hg38/hg38.fa"
    args.output_path = "/mnt/storage/data/joana_pales/msc-joana/data/pretrain/val_pretrain_data"
    args.kmer = 6
    args.val_sampling = 26 # per validation n'agafo 1 cada 26


    Process(args)


if __name__ == "__main__":
    main()
