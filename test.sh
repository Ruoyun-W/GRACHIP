#!/bin/bash

python prediction.py \
	--chrom "chr2" \
	--start 196760000 \
	--genomic_data_paths "data/bwlist.txt" \
	--output "./output" \
	--genome_path "./data/hg38.fa" 