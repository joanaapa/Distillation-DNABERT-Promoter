"""
Construct the promoter dataset. Adapted from DNABERT
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
from data.pretrain.process_pretrain_data import get_kmer_sentence
from utils import read_txt

max_length = 0



def write_file(lines, path, kmer, head=True, seq_index=0, label_index=1):
    """
    It takes a list of lists, where each list is a line of a file, and writes it to a file
    
    :param lines: the lines to write to the file
    :param path: the path to the file you want to write to
    :param kmer: the length of the k-mer. If kmer is 0, then the whole sequence is used
    :param head: whether to write the header or not, defaults to True (optional)
    :param seq_index: the index of the sequence in the line, defaults to 0 (optional)
    :param label_index: the index of the label in the line, defaults to 1 (optional)
    """
    with open(path, "wt") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        if head:
            tsv_w.writerow(["sentence", "label"])
        for line in lines:
            if kmer == 0:
                sentence = str(line[seq_index])
            else:
                sentence = str(get_kmer_sentence("".join(line[seq_index].split()), kmer))
            if label_index is None:
                label = "0"
            else:
                label = str(line[label_index])
            tsv_w.writerow([sentence, label])


def Generate_prom_train_dev(args):
    # read TATA and noTATA files + add labels
    tata_neg = args.file_path + "/TATAprom_neg.txt"
    tata_pos = args.file_path + "/TATAprom_pos.txt"
    notata_neg = args.file_path + "/nonTATAprom_neg.txt"
    notata_pos = args.file_path + "/nonTATAprom_pos.txt"

    tata = read_txt(tata_neg)
    n_tata_neg = len(tata)
    tata = tata + read_txt(tata_pos)
    tata_lbl = [0] * n_tata_neg + [1] * (len(tata) - n_tata_neg)
    tata = [list(x) for x in zip(tata, tata_lbl)]
    write_file(tata[n_tata_neg:], args.output_path + '/' + str(args.kmer) + "kmer/tata.tsv", args.kmer)

    notata = read_txt(notata_neg)
    n_notata_neg = len(notata)
    notata = notata + read_txt(notata_pos)
    notata_lbl = [0] * n_notata_neg + [1] * (len(notata) - n_notata_neg)
    notata = [list(x) for x in zip(notata, notata_lbl)]
    write_file(notata[n_notata_neg:], args.output_path + '/' + str(args.kmer) + "kmer/notata.tsv", args.kmer)

    # shuffle all the data and split them
    random.shuffle(tata)
    random.shuffle(notata)
    num_tata_test = int(len(tata) * float(args.valtest_percentage))
    tata_test_lines = tata[:num_tata_test]
    num_notata_test = int(len(notata) * float(args.valtest_percentage))
    notata_test_lines = notata[:num_notata_test]
    train_lines = tata[2*num_tata_test:] + notata[2*num_notata_test:]
    val_lines = tata[num_tata_test:2*num_tata_test] + notata[num_tata_test:2*num_notata_test]
    test_lines = tata_test_lines + notata_test_lines
    
    # Save the files
    write_file(train_lines, args.output_path + '/' + str(args.kmer) + "kmer/train.tsv", args.kmer)
    write_file(val_lines, args.output_path + '/' + str(args.kmer) + "kmer/dev.tsv", args.kmer)
    write_file(test_lines, args.output_path + '/' + str(args.kmer) + "kmer/test.tsv", args.kmer)


    # To later evaluate performance on only tata/notata
    write_file(tata_test_lines, args.file_path+"/tata_dev.tsv", args.kmer)
    write_file(notata_test_lines, args.file_path+"/notata_dev.tsv", args.kmer)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--kmer", default=4, type=int, help="K-mer",)
    parser.add_argument("--file_path", default="/workspace/data/promoters", type=str, help="The path of the file to be processed",)
    parser.add_argument("--output_path", default="/workspace/data/promoters", type=str, help="The path of the processed data",)
    parser.add_argument("--valtest_percentage", default=0.1, type=str, help="Percentage of data used for val and for test (same for both)",)


    # Override arg parser for debugging
    sys.argv = ['process_finetune_data.py', '--kmer', '6', '--file_path', '/mnt/storage/data/joana_pales/msc-joana/data/promoters','--output_path', '/mnt/storage/data/joana_pales/msc-joana/data/promoters', "--valtest_percentage", "0.1"]

    args = parser.parse_args()

    Generate_prom_train_dev(args)


if __name__ == "__main__":
    main()
