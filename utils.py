import numpy as np
from pyfaidx import Fasta
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def fasta2list(fasta_file):

    data = Fasta(fasta_file)

    sequences = []
    for promoter_id in data.keys():
        seq = data[promoter_id][:].seq
        sequences.append(seq)
    return sequences


def save_txt(seq_list, new_filepath):
    with open(new_filepath, "w") as f:
        f.write("\n".join(seq_list))


def read_txt(filepath):
    with open(filepath) as f:
        seq_list = f.read().splitlines()
    return seq_list
