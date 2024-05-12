import numpy as np
import cooler
import torch
import joblib
import torch_geometric.data
import pandas as pd

def fetch_interaction_data(input_edge_path, df_region):
    """
    Fetches interaction data from a cooler file for specified genomic regions.

    Parameters:
    - input_edge_path: Path to the cooler file.
    - df_region: DataFrame with at least 'chrom', 'start', and 'end' columns for one region.

    Returns:
    - interaction_data: Array with columns for the indices of interacting bins and their interaction values.
    """
    # Constants
    MAXNUM = 100  # Define a maximum number after which to cap the interaction values
    NORMALIZATION_FACTOR = 20000  # Normalization factor for the log transform

    # Load the cooler file
    c1 = cooler.Cooler(input_edge_path)

    # Extract the region of interest details
    chrom = df_region.loc[0, 'chrom']
    start = df_region['start'].min()
    end = df_region['end'].max()

    # Fetch the unbalanced interaction matrix
    matrix = c1.matrix(balance=False, sparse=True).fetch((chrom, start, end))

    # Fetch bias weights
    bias = c1.bins().fetch((chrom, start, end))['weight'].values

    # Apply balancing weights to the sparse matrix data
    # valid_indices = np.isfinite(bias)  # Filter to avoid NaNs in calculations
    # valid_bias = bias[valid_indices]
    # matrix = matrix.tocoo()  # Ensure matrix is in COOrdinate format for easier manipulation
    matrix.data = bias[matrix.row] * bias[matrix.col] * matrix.data

    # Convert to dense matrix and apply transformations
    arr = np.nan_to_num(matrix.toarray())
    transformed_arr = np.log(arr * NORMALIZATION_FACTOR + 1)
    transformed_arr[transformed_arr > MAXNUM] = MAXNUM

    # Extract indices and values for interactions above the threshold
    indices_matrix = np.where(transformed_arr > 0)
    values_edge = transformed_arr[indices_matrix]
    index1_edge, index2_edge = indices_matrix[0].flatten(), indices_matrix[1].flatten()
    values_edge = values_edge.flatten()

    # Stack the indices and values into a single array
    interaction_data = np.column_stack((index1_edge, index2_edge, values_edge))

    return interaction_data



def create_one_data(feature_matrix, input_edge_path, encoded_DNA, df_region):
    """
    Creates a single data object for graph-based models integrating genomic and DNA features along with interaction data.

    Parameters:
    - feature_matrix: 2D array or DataFrame containing genomic features.
    - input_edge_path: Path to interaction data, which includes the 'matrix'.
    - encoded_DNA: 2D array with encoded DNA data.
    - df_region: DataFrame containing the regions data (chromosome information).

    Returns:
    - data: A PyTorch Geometric data object with genomic and DNA features along with interaction edges and attributes.
    """
    # Load the scaler and standardize the DNA data
    scaler = joblib.load(f'scalers/{df_region["chrom"][0]}_10kscaler.joblib')
    standardized_DNA = scaler.transform(encoded_DNA)

    # Fetch interaction data
    interaction_data = fetch_interaction_data(input_edge_path, df_region)  # Assumed corrected fetching method

    # Prepare edges for the graph
    edge_index = torch.tensor([item[:2] for item in interaction_data], dtype=torch.long).t()
    edge_attr = torch.tensor([item[2] for item in interaction_data], dtype=torch.float)

    # Assuming the number of nodes is equal to the number of rows in feature_matrix
    num_nodes = feature_matrix.shape[0] if hasattr(feature_matrix, 'shape') else len(feature_matrix)
    # Create the PyTorch Geometric data object
    data = torch_geometric.data.Data(
        x1=torch.tensor(feature_matrix.values, dtype=torch.float),
        x2=torch.tensor(standardized_DNA, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )

    return data    