"""
Generate the negative sequences for promoter dataset
"""

import itertools
import random
import re

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from utils import read_txt, save_txt


def get_neg_seq(pos_seq, num_part=20, keep=8):
    """
    Split the positive sequence into 20 parts, and randomly select 8 parts to keep. The remaining
    12 parts are replaced by random sequences. Adapted from https://github.com/egochao/DeePromoter
    
    :param pos_seq: the positive sequence
    :param num_part: the number of parts to split the sequence into, defaults to 20 (optional)
    :param keep: the number of parts of the sequence to keep, defaults to 8 (optional)
    :return: A string of the same length as the input sequence, but with some of the bases randomly
    replaced.
    """

    length = len(pos_seq)
    part_len = length // num_part
    if part_len * num_part < length:
        num_part += 1

    iterator = np.arange(num_part)
    keep_parts = random.sample(list(iterator), k=keep)

    outpro = list()
    for it in iterator:
        start = it * part_len
        pro_part = pos_seq[start : start + part_len]
        if it in keep_parts:
            outpro.extend(pro_part)
        else:
            pro_part = random.choices(["A", "C", "G", "T"], k=len(pro_part))
            outpro.extend(pro_part)

    return "".join(outpro)


def generate_neg(pos_data):
    """
    For each sequence in the positive data, generate a negative sequence by randomly shuffling the
    positive sequence
    
    :param pos_data: a list of positive sequences
    :return: A list of negative sequences.
    """
    neg_data = []
    for pos_seq in pos_data:
        neg_seq = get_neg_seq(pos_seq)
        neg_data.append(neg_seq)
    return neg_data


def find_tata(ref_genome, promoter_df):
    """
    Return coordinates of 'TATA' seq in ref genome that are not promoters
    
    :param ref_genome: a dictionary of chromosomes from the reference genome
    :param promoter_df: a dataframe
    :return: A list of lists, where each list contains the chromosome and the location of the TATA
    sequence.
    """
    

    print("Getting TATA patterns locations...")

    promoter_df = promoter_df[["Chr", "TSS"]]
    coord = []
    # For each chromosome, take sequences in-between promoters and find TATA sequences
    for i in range(1, 23):
        chrx = ref_genome["chr" + str(i)][:].seq
        tss_chr = [0] + promoter_df[promoter_df["Chr"] == "chr" + str(i)]["TSS"].values.tolist()
        finds = []
        for j in range(len(tss_chr) - 1):
            seq = chrx[tss_chr[j] + 50 : tss_chr[j + 1] - 250]
            if len(seq) > 500:
                new_finds = [m.start(0) + tss_chr[j] + 50 for m in re.finditer("tata", seq, re.I)]
                finds = finds + new_finds
        coord += [list(x) for x in zip(["chr" + str(i)] * len(finds), finds)]
    return coord


def extract_seq(ref_genome, chrx, start, end):
    return ref_genome[chrx][start:end].seq.upper()


# Generate negative sequences of nonTATA promoters following DeePromoter methodology
pos_txt = "/mnt/storage/data/joana_pales/msc-joana/data/promoters/nonTATAprom_pos.txt"
pos_data = read_txt(pos_txt)
neg_data = generate_neg(pos_data)
filepath = "/mnt/storage/data/joana_pales/msc-joana/data/promoters/nonTATAprom_neg.txt"
save_txt(neg_data, filepath)


# Generate negative sequences of TATA promoters by taking non-promoter TATA sequences
filepath = "/mnt/storage/data/joana_pales/msc-joana/data/hg38/hg38.fa"
ref_genome = Fasta(filepath)
bed_file = "/mnt/storage/data/joana_pales/msc-joana/data/raw/promoters_coord.bed"
promoter_df = pd.read_csv(bed_file, sep="\t", header=None, names=["Chr", "TSS", "Location", "V", "Name", "Strand"])

n_tata = 3065
tata_locations = find_tata(ref_genome, promoter_df)
chosen = random.sample(tata_locations, n_tata)
shift = random.choices(range(20), k=n_tata)  # To avoid that all the TATA end up at exactly -25bp
tata_neg = [extract_seq(ref_genome, x[0], x[1] - 215 + shift[i], x[1] + 65 + shift[i]) for i, x in enumerate(chosen)]
save_txt(tata_neg, "/mnt/storage/data/joana_pales/msc-joana/data/promoters/TATAprom_neg.txt")
