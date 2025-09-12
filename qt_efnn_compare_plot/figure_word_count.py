import numpy as np
from scipy.interpolate import pade


w=np.array(
    [24.68,25.85,39.47,26.07,43.4]
)
h=np.array(
    [22.10,24.83,16.67,25.35,23.96]
)
columns=np.array(
    [1,1,2,1,1]
)


def calculate_prl_word_count_arrays(w, h, columns):
    """
    Calculate PRL word count for figures using numpy arrays.

    Args:
        w: numpy array of widths
        h: numpy array of heights
        columns: numpy array of column numbers (1 or 2)

    Returns:
        results: numpy array of word counts for each figure
        total: total word count
    """

    # Calculate aspect ratios
    aspect_ratios = w / h

    # Calculate word counts based on column layout
    # Single column: (150 / aspect_ratio) + 20
    # Two column: (300 / (0.5 * aspect_ratio)) + 40

    word_counts = np.where(
        columns == 1,
        (150 / aspect_ratios) + 20,  # Single column formula
        (300 / (0.5 * aspect_ratios)) + 40  # Two column formula
    )

    # Round to nearest integer
    word_counts = np.round(word_counts).astype(int)

    return word_counts, np.sum(word_counts)


# Calculate word counts
word_counts, total_words = calculate_prl_word_count_arrays(w, h, columns)

# Print results
print("Figure Word Count Analysis")
print("=" * 60)
print("Fig | Width  | Height | Aspect | Cols | Words")
print("-" * 60)

for i in range(len(w)):
    aspect_ratio = w[i] / h[i]
    print(f" {i+1:2d} | {w[i]:6.2f} | {h[i]:6.2f} | {aspect_ratio:6.3f} | "
          f"  {columns[i]:1d}  | {word_counts[i]:5d}")

print("=" * 60)
print(f"Total word count for all figures: {total_words} words")

# Individual results as arrays
print(f"\nWord counts array: {word_counts}")
print(f"Aspect ratios: {np.round(w/h, 3)}")