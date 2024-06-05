import pyBigWig
import pandas as pd
import numpy as np
def create_region_dataframe(chrom, start, sample_size, bin_size=10000):
    """
    Creates a DataFrame representing genomic regions.

    Parameters:
    - chrom: Chromosome identifier (e.g., 'chr1').
    - start: Initial start position for the regions (integer).
    - sample_size: Number of bins to create.
    - bin_size: Size of each bin in base pairs (default is 10,000).

    Returns:
    - DataFrame with columns 'chrom', 'start', 'end'.
    """
    # Calculate the region to fetch
    region_start = start // bin_size * bin_size
    region_end = start + sample_size * bin_size

    # Create a DataFrame for the regions
    data = {
        'chrom': [chrom] * sample_size,
        'start': [region_start + i * bin_size for i in range(sample_size)],
        'end': [region_start + (i + 1) * bin_size for i in range(sample_size)]
    }
    df_regions = pd.DataFrame(data)
    return df_regions

def fetch_values(row, bw, sample_size):
    region = (row['chrom'], int(row['start']), int(row['end']))
    values = bw.values(*region)

    values=np.nan_to_num(values)
    mean_values=[]
    for i in range(len(values)//sample_size):
        mean_values.append(np.mean(values[i*sample_size:(i+1)*sample_size]))
    # transformed_value = np.log(np.max(mean_values)+1)
    transformed_value = np.max(values)
    return transformed_value

def fetch_genomic_data(genomic_data_paths, df_regions):
    """
    Fetch genomic data from BigWig files for a specified region.

    Parameters:
    - genomic_data_paths: Path to a text file containing paths to BigWig files.
    - start: Start position of the region of interest.
    - bin_size: Size of the bins for averaging genomic signals.
    - sample_size: Total size of the window to consider after the start position.

    Returns:
    - A DataFrame with genomic signals from various BigWig files.
    """
    sample_size = df_regions.shape[0]
    # Read the list of BigWig files from the provided path
    df_genomic = pd.DataFrame(index=df_regions.index)
    with open(genomic_data_paths, 'r') as file:
        bw_paths = [line.strip() for line in file]

    # Loop over each BigWig file to extract genomic data
    ind=0
    for bw_file in bw_paths:
        if bw_file and bw_file.upper() != 'NA':  # Check if the file path is valid and not 'NA'
            try:
                bw = pyBigWig.open(bw_file)
                df_genomic[bw_file] = df_regions.apply(fetch_values, axis=1, args=(bw, sample_size))
                bw.close()
            except Exception as e:
                print(f"Failed to process {bw_file}: {e}")
                df_genomic[bw_file] = 0  # Fill with zeros in case of failure to read the file
        else:
            # Fill the column with zeros if the file path is 'NA' or invalid
            bw_file=bw_file+str(ind)
            ind+=1
            df_genomic[bw_file] = 0

    return df_genomic