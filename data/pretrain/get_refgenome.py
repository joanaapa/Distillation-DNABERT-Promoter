"""
Download reference genome
"""

import sys
import wget


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
wget.download(url, "/mnt/storage/data/joana_pales/msc-joana/data/hg38/hg38.fa.gz", bar=bar_progress)
print("done")
