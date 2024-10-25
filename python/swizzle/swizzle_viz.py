import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import argparse
import imageio
import os
import shutil

# Create an index matrix to track colors for each element (used for color visualization only)
def create_index_matrix(matrix):
    num_rows, num_cols = matrix.shape
    index_matrix = np.tile(np.arange(num_cols), (num_rows, 1))  # Create a matrix where each column has its index
    return index_matrix


# Create a colormap with exactly 32 distinct colors
def create_32_color_palette(cols, cap=32, repeat_if_exceed=False):
    cmap = plt.get_cmap('tab20')  # Start with 'tab20' for the first 20 colors
    extra_colors = sns.color_palette("Paired", 12)  # 'Paired' provides 12 additional distinct colors
    colors = [cmap(i % 20) for i in range(20)] + extra_colors  # Combine the 20 from 'tab20' and 12 from 'Paired'

    # Cap the colors at the specified limit
    capped_colors = colors[:cap]

    # If repeat_if_exceed is True, repeat the colors if the requested cap exceeds the number of available colors
    if repeat_if_exceed:
        fn = lambda n: capped_colors * (n // cap) + capped_colors[:n % cap] if n > cap else capped_colors[:n]
        capped_colors = fn(cols)
    
    return capped_colors


# Function to visualize both the data matrix and index matrix side by side with dynamic sizing
def visualize_matrices(matrix, index_matrix, colors, step, output_dir):
    num_rows, num_cols = matrix.shape

    # Dynamically adjust figure size based on matrix dimensions (rows and cols)
    fig_size_factor = max(num_rows, num_cols) / 5  # Scale the figure size
    fig_size_factor = max(fig_size_factor, 6)  # Set a minimum figure size to prevent too much zoom for small matrices

    fig, axes = plt.subplots(1, 2, figsize=(fig_size_factor * 6, fig_size_factor * 3))  # Adjust figure size

    # Set text size with a minimum threshold to prevent it from getting too small
    text_size = max(10, 25 - int(num_cols / 2))  # Set minimum text size at 10

    # Visualize the data matrix
    ax1 = axes[0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax1.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[index_matrix[i, j]], ec='black'))
            ax1.text(j + 0.5, i + 0.5, str(matrix[i, j]), ha='center', va='center', color='black', fontsize=text_size)
    ax1.set_xlim(0, matrix.shape[1])
    ax1.set_ylim(0, matrix.shape[0])
    ax1.invert_yaxis()
    ax1.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax1.set_xticklabels([f"Col {i}" for i in range(matrix.shape[1])], fontsize=text_size)
    ax1.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax1.set_yticklabels([f"Row {i}" for i in range(matrix.shape[0])], fontsize=text_size)
    ax1.set_title(f"Data Matrix (Step {step})", fontsize=text_size + 4)
    ax1.set_xlabel("Columns", fontsize=text_size)
    ax1.set_ylabel("Rows", fontsize=text_size)
    ax1.grid(False)

    # Visualize the index matrix
    ax2 = axes[1]
    for i in range(index_matrix.shape[0]):
        for j in range(index_matrix.shape[1]):
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[index_matrix[i, j]], ec='black'))
            ax2.text(j + 0.5, i + 0.5, str(index_matrix[i, j]), ha='center', va='center', color='black', fontsize=text_size)
    ax2.set_xlim(0, index_matrix.shape[1])
    ax2.set_ylim(0, index_matrix.shape[0])
    ax2.invert_yaxis()
    ax2.set_xticks(np.arange(index_matrix.shape[1]) + 0.5)
    ax2.set_xticklabels([f"Col {i}" for i in range(index_matrix.shape[1])], fontsize=text_size)
    ax2.set_yticks(np.arange(index_matrix.shape[0]) + 0.5)
    ax2.set_yticklabels([f"Row {i}" for i in range(index_matrix.shape[0])], fontsize=text_size)
    ax2.set_title(f"Index Matrix (Step {step})", fontsize=text_size + 4)
    ax2.set_xlabel("Columns", fontsize=text_size)
    ax2.set_ylabel("Rows", fontsize=text_size)
    ax2.grid(False)

    # Save the figure as an image
    filename = os.path.join(output_dir, f"step_{step}.png")
    plt.savefig(filename)
    plt.close()
    
# Function to create an XOR matrix based on row and column indices
def create_xor_matrix(matrix):
    num_rows, num_cols = matrix.shape
    xor_matrix = np.zeros((num_rows, num_cols), dtype=int)

    for row in range(num_rows):
        for col in range(num_cols):
            xor_matrix[row, col] = row ^ col  # XOR the row and column indices

    return xor_matrix

# Function to create a new matrix and index matrix, and generate images for GIF
def create_new_matrix_and_gif(matrix, xor_matrix, index_matrix, output_dir, colors):
    num_rows, num_cols = matrix.shape
    new_matrix = np.copy(matrix)  # Make a copy of the original matrix
    new_index_matrix = np.copy(index_matrix)  # Make a copy of the original index matrix

    # Create an image for each step (row processing)
    for row in range(num_rows):
        for col in range(num_cols):
            xor_result = xor_matrix[row, col]

            # Handle out-of-bounds XOR results by setting the value to -1
            if xor_result >= num_cols:
                new_matrix[row, col] = -1
                new_index_matrix[row, col] = -1  # Set index to -1 for out-of-bounds
            elif xor_result != col:  # Swap if XOR result is valid
                new_matrix[row, col] = matrix[row, xor_result]
                new_index_matrix[row, col] = index_matrix[row, xor_result]

        # Visualize and save the matrices after processing each row
        visualize_matrices(new_matrix, new_index_matrix, colors, step=row, output_dir=output_dir)

# Function to generate a GIF from saved images
def create_gif(output_dir, gif_filename, duration=1.0):
    images = []
    for i in range(len(os.listdir(output_dir))):
        filename = os.path.join(output_dir, f"step_{i}.png")
        images.append(imageio.imread(filename))
    
    # Create the GIF with adjustable duration between frames
    imageio.mimsave(gif_filename, images, fps=duration)

# Function to generate a random matrix for testing
def generate_matrix(rows, cols):
    return np.arange(0, cols * rows).reshape(rows, cols)

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Matrix visualization tool with XOR operation row by row")
    
    # Add arguments for matrix size
    parser.add_argument('--rows', type=int, default=4, help='Number of rows in the matrix')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the matrix')
    parser.add_argument('--output-dir', type=str, default='output_images', help='Directory to save images for the GIF')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration of each frame in seconds for the GIF')
    parser.add_argument('--color-cap', type=int, default=32, help='Caps the number of colors')
    parser.add_argument('--color-repeat', action="store_true", help='repeats color pallete')
    
    args = parser.parse_args()
    args.gif = f"matrix-xor-{args.rows}x{args.cols}.gif"

    # Create output directory if it doesn't exist
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)

    # Generate a random matrix of specified size
    matrix = generate_matrix(args.rows, args.cols)
    index_matrix = create_index_matrix(matrix)

    # Create the XOR matrix
    xor_matrix = create_xor_matrix(matrix)
    colors = create_32_color_palette(args.cols, args.color_cap, args.color_repeat)
    
    # Create a new matrix and generate images for the GIF
    create_new_matrix_and_gif(matrix, xor_matrix, index_matrix, args.output_dir, colors)

    # Create the GIF from the saved images
    create_gif(args.output_dir, args.gif, duration=args.duration)

    print(f"GIF created: {args.gif}")

if __name__ == '__main__':
    main()
