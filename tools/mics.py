import torch
from model.grachip import *

def average_symmetric(matrix):
    """
    Average a matrix with its transpose to enforce symmetry.
    Parameters:
    - matrix: A 2D tensor.
    Returns:
    - sym_matrix: A symmetric 2D tensor.
    """
    return 0.5 * (matrix + matrix.T)

def cap_values(matrix, N):
    """
    Cap the values in a matrix to a maximum specified value.
    Parameters:
    - matrix: Tensor, the matrix to be capped.
    - N: float, the maximum value for elements in the matrix.
    Returns:
    - capped_matrix: Tensor, the matrix with values capped.
    """
    return torch.clamp(matrix, max=N)

def predict(model_path, data, device, save_path=None):
    """
    Load a pre-trained model and make predictions on provided data.
    Parameters:
    - model_path: str, path to the pre-trained model file.
    - data: Tensor, the input data for making predictions.
    - device: str, the compute device ('cuda' or 'cpu').
    - save_path: str, optional path to save the predictions.
    Returns:
    - pred: Tensor, the predicted matrix after post-processing.
    """
    # Define model configuration
    samplesize = 200 
    
    # Load the trained model from disk
    model = torch.load(model_path)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Assuming 'data' is already on the correct device and in the correct format
        output = model(data.to(device), batchSize=1, eval=True)  # Obtain output from model
        pred = output[0]  # Assume the model returns the prediction as the first element

        # Reshape, symmetrize, and cap the prediction matrix
        pred = pred.detach().cpu().view(samplesize, samplesize)
        pred = average_symmetric(pred)  # Make the matrix symmetric
        pred = cap_values(pred, 7.2)  # Cap values at 7.2

        # Optionally save the prediction
        if save_path:
            torch.save(pred, save_path)

    return pred
