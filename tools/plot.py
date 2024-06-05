import matplotlib.pyplot as plt
def white_to_red_colormap(steps=256):
    # Start with white and linearly decrease green and blue to get to red
    return [(1, g / steps, b / steps) for g, b in zip(range(steps, 0, -1), range(steps, 0, -1))]

def plot(image_data,text=None,y=None, cmap=plt.cm.colors.ListedColormap(white_to_red_colormap()), hide_axes=True, border_color='black', border_width=1, save_path=None,vmax=None):
    """
    Create a plot with optional axis hiding and a border.

    Args:
        image_data (numpy.ndarray): Image data to be plotted.
        cmap (str, optional): Colormap for the image. Default is 'viridis'.
        hide_axes (bool, optional): Whether to hide x and y axes. Default is True.
        border_color (str, optional): Color of the border. Default is 'black'.
        border_width (int, optional): Width of the border lines. Default is 1.
        save_path (str, optional): Path to save the image (e.g., 'output.png'). Default is None (not saving).

    Returns:
        None
    """
    plt.figure(figsize=(4,3))
    if text:
        plt.title(text,fontsize=10)

    # Plot the image
    if vmax:
        plt.imshow(image_data, cmap=cmap,vmax=vmax)
    else :
        plt.imshow(image_data, cmap=cmap)
    plt.ylabel(y)
    # Customize axis appearance (hide them if requested)
    if hide_axes:
        ax = plt.gca()
        plt.tick_params(axis='both', which='both', length=0)
        plt.xticks([])
        plt.yticks([])
        
    # plt.colorbar()  # Add a color bar with a label
    # Adjust the position of the color bar
    # plt.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the color bar
    # Save the image if a save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    # plt.show()