Generate promoter dataset.
1. TATA and non-TATA promoters downloaded from EPDnew database
2. get_promoter.py used to generate negative sequences [available txt files: nonTATAprom_neg.txt, nonTATAprom_pos.txt, TATAprom_neg.txt, TATA prom_pos.txt]
3. process_finetune_data.py used to split data in k-mer and save train/test/dev sets plus tata/nontata [available for 6mer: tata.tsv, test.tsv]
