import subprocess
import json
import numpy as np
import os

def get_DNA_hidden_state(dnabert_path, kmer_dna_path):
    """
    Executes a Python script to generate DNA hidden states using a DNABERT model.

    Parameters:
    - dnabert_path: Path to the DNABERT model.
    - kmer_dna_path: Path to DNA k-mer sequence or path to DNA sequence data.

    Returns:
    - DNA_matrix: NumPy array of DNA hidden states as outputted by the DNABERT model.
    """
    # Define the command to run the external Python script
    command = [
        'python3', 'tools/DNA_encoder.py',
        '--model_type', 'dna',
        '--tokenizer_name', f'dna6',  
        '--model_name_or_path', dnabert_path,
        '--task_name', 'dnaprom',
        '--do_predict',
        '--data_dir', kmer_dna_path,  
        '--max_seq_length', '75',
        '--per_gpu_pred_batch_size', '128',
        '--output_dir', dnabert_path,
        '--predict_dir', os.path.join(kmer_dna_path,"output.npy"),
        '--n_process', '2'
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if the command was successful
    if result.returncode != 0:
        print("Error in executing DNABERT script:")
        print(result.stderr)
        return None
    
    # Assuming the output is saved to a numpy file
    try:
        output_file_path = os.path.join(kmer_dna_path,"output.npy")
        # Load the numpy file containing the DNA_matrix
        dna_matrix = np.load(output_file_path)
        os.remove(output_file_path)
        os.remove(os.path.join(kmer_dna_path,"train.tsv"))
        os.remove(os.path.join(kmer_dna_path,"cached_train_75_dnaprom"))

        return dna_matrix
    except Exception as e:
        print(f"Failed to load DNA matrix: {e}")
        return None