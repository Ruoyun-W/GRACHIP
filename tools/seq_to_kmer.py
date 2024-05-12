import os
import pandas as pd
import numpy as np
import sys
import pyfaidx
from pyfaidx import Fasta

k=6

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def regions2kmers(bed_df, genome_path, k, output_dir):
    """
    Function to extract k-mers from a DataFrame of BED format regions.
    
    Parameters:
    - bed_df: DataFrame with columns ['chromosome', 'start', 'end']
    - genome_path: Path to the reference genome file.
    - k: Length of the k-mers to extract.
    - output_dir: Directory to save the output files.
    """
    # Load the reference genome
    genome = Fasta(genome_path)

    # Prepare the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each region and extract sequences
    kmers = []
    for index, row in bed_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        seq = genome[chrom][start:end].seq.upper()
        kmers.append(seq2kmer(seq, k))

    # Create a DataFrame for k-mers and save it
    df = pd.DataFrame(kmers, columns=["sample"])
    df['label'] = 0
    output_file = os.path.join(output_dir, "train.tsv")
    
    df.to_csv(output_file, sep="\t", index=False, header=True)

    return output_dir

