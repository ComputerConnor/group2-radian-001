"""
Maximum Subarray Experimental Suite
CSC 2400 - Design & Analysis of Algorithms (Fall 2025)

HOW TO RUN:
-----------------------------------
> python main.py

OUTPUTS:
- Formatted runtime tables printed to console
- CSV files for each algorithm:
      kadane_results.csv
      dc_results.csv
      dp2d_results.csv
- Matplotlib runtime graphs:
      kadane_runtime.png
      dc_runtime.png
      dp2d_runtime.png

These files can be used directly in the project report and slides.
-----------------------------------
"""

import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# ============================================================
# SYNTHETIC DATASET GENERATION
# These functions produce artificial biological-style data:
# 1D data simulates gene expression curves or protein levels.
# 2D data simulates spatial/genomic matrices such as Hi-C contact maps.
# ============================================================

def generate_1d_signal(n: int, noise_level=1.0, signal_range=(10, 50)):
    """
    Generates a 1D synthetic time-series with Gaussian noise and
    an implanted high-value region (a "biological hotspot").
    
    Parameters:
        n (int): Number of elements
        noise_level (float): Standard deviation of noise
        signal_range (tuple): Length range for the signal peak
    
    Returns:
        np.array: Generated 1D array
    """
    data = np.random.normal(0, noise_level, n)
    start = np.random.randint(0, n // 2)
    length = np.random.randint(*signal_range)
    data[start:start + length] += np.random.uniform(5, 10)
    return data


def generate_2d_signal(n: int, m: int, noise_level=1.0, signal_shape=(10, 15)):
    """
    Generates a 2D synthetic matrix with Gaussian noise and an implanted
    high-value rectangular region. This mimics a hotspot in Hi-C matrices.
    
    Parameters:
        n (int): Rows
        m (int): Columns
        noise_level (float): Standard deviation of noise
        signal_shape (tuple): Height x Width of the hotspot

    Returns:
        np.array: Generated 2D matrix
    """
    dataset = np.random.normal(0, noise_level, (n, m))
    i = np.random.randint(0, n - signal_shape[0])
    j = np.random.randint(0, m - signal_shape[1])
    dataset[i:i + signal_shape[0], j:j + signal_shape[1]] += np.random.uniform(5, 10)
    return dataset


# ============================================================
# DIVIDE AND CONQUER MAX SUBARRAY
# Classical O(n log n) divide-and-conquer solution:
# Recursively solves left, right, and crossing subarrays.
# ============================================================

def max_crossing_sum(array, l, m, h):
    """
    Computes maximum sum of a subarray crossing the midpoint.
    Required for the combine step of divide-and-conquer.

    Parameters:
        array: Input array
        l: Left index
        m: Middle index
        h: Right index

    Returns:
        int/float: Maximum crossing subarray sum
    """
    # Max subarray ending at mid going left
    left_sum = float('-inf')
    total = 0
    for i in range(m, l - 1, -1):
        total += array[i]
        left_sum = max(left_sum, total)

    # Max subarray starting at mid+1 going right
    right_sum = float('-inf')
    total = 0
    for i in range(m + 1, h + 1):
        total += array[i]
        right_sum = max(right_sum, total)

    return left_sum + right_sum


def max_subarray_dc(array, l, h):
    """
    Recursive divide-and-conquer function to compute
    the maximum subarray sum.

    Returns:
        Maximum subarray sum in array[l..h]
    """
    if l == h:
        return array[l]

    m = (l + h) // 2

    # Maximum of left half, right half, and crossing segment
    return max(
        max_subarray_dc(array, l, m),
        max_subarray_dc(array, m + 1, h),
        max_crossing_sum(array, l, m, h)
    )


def max_subarray_sum(array):
    """ Helper wrapper to simplify calling DC algorithm. """
    return max_subarray_dc(array, 0, len(array) - 1)


# ============================================================
# KADANE'S ALGORITHM
# Optimal O(n) greedy algorithm for 1D maximum subarray.
# Tracks the best subarray ending at each index.
# ============================================================

def max_subarray_kadanes(array):
    """
    Kadane's Algorithm (O(n))
    Computes the maximum subarray sum in a single pass.
    """
    max_sum = array[0]
    current_sum = array[0]
    for x in array[1:]:
        current_sum = max(x, current_sum + x)
        max_sum = max(max_sum, current_sum)
    return max_sum


# ============================================================
# 2D DP MAX SUBMATRIX
# Uses row-collapsing + Kadane to compute max submatrix.
# This method runs in O(n^3).
# ============================================================

def max_submatrix(mat):
    """
    Computes the maximum-sum submatrix of a 2D matrix using the 2D Kadane approach:
    Fix left column -> expand right column -> collapse rows -> run Kadane on 1D array.

    Parameters:
        mat: 2D list or numpy array

    Returns:
        float: Maximum submatrix sum
    """
    rows = len(mat)
    cols = len(mat[0])
    max_sum = float('-inf')

    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += mat[row][right]
            max_sum = max(max_sum, max_subarray_kadanes(temp))

    return max_sum


# ============================================================
# EXPERIMENT RUNNER + UTILITIES
# These functions automate testing each algorithm across
# multiple input sizes, repeat trials, compute averages,
# print formatted tables, generate CSVs, and save graphs.
# ============================================================

def run_experiment_1d(algorithm, sizes, repeats=10, name="algorithm"):
    """
    Runs benchmarking for 1D algorithms (Kadane and D&C).
    Repeats each input size multiple times and averages runtime.

    Outputs:
        - Console table
        - CSV results file
        - Runtime plot PNG
    """
    results = []  # (size, avg_time)

    print(f"\n=== {name.upper()} (1D) RESULTS ===")
    print("Size        Average Time (s)")
    print("---------------------------------")

    for size in sizes:
        times = []
        for _ in range(repeats):
            arr = generate_1d_signal(size)
            start = time.perf_counter()
            algorithm(arr)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / repeats
        results.append((size, avg_time))
        print(f"{size:<12}{avg_time:.6f}")

    save_csv(f"{name}_results.csv", results)
    save_plot(f"{name}_runtime.png", results, title=f"{name.upper()} Runtime (1D)")

    return results


def run_experiment_2d(sizes, repeats=5, name="dp2d"):
    """
    Runs benchmarking for the 2D DP maximum submatrix algorithm.

    Outputs:
        - Console table
        - CSV results file
        - Runtime plot PNG
    """
    results = []  # (matrix_size, avg_time)

    print(f"\n=== {name.upper()} (2D DP) RESULTS ===")
    print("Matrix Size        Average Time (s)")
    print("-------------------------------------")

    for (n, m) in sizes:
        times = []
        for _ in range(repeats):
            mat = generate_2d_signal(n, m)
            start = time.perf_counter()
            max_submatrix(mat)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / repeats
        results.append((f"{n}x{m}", avg_time))
        print(f"{n}x{m:<15}{avg_time:.6f}")

    save_csv(f"{name}_results.csv", results)
    save_plot_2d(f"{name}_runtime.png", results, title="2D DP Runtime")

    return results


# ============================================================
# CSV + PLOT UTILITIES
# Writing results to CSV and saving Matplotlib graphs.
# ============================================================

def save_csv(filename, rows):
    """ Saves results (size, avg_time) into a CSV file. """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InputSize", "AvgTime"])
        writer.writerows(rows)


def save_plot(filename, results, title="Runtime"):
    """ Saves line chart of runtime vs input size for 1D algorithms. """
    sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.plot(sizes, times, marker='o')
    plt.title(title)
    plt.xlabel("Input Size")
    plt.ylabel("Avg Runtime (s)")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_plot_2d(filename, results, title="Runtime"):
    """ Saves runtime chart for 2D DP (x-axis is matrix size). """
    sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.plot(times, marker='o')
    plt.xticks(range(len(sizes)), sizes)
    plt.title(title)
    plt.xlabel("Matrix Size")
    plt.ylabel("Avg Runtime (s)")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# ============================================================
# MAIN EXECUTION
# Runs all experiments when executed from the command line.
# ============================================================

if __name__ == "__main__":
    sizes_1d = [1000, 5000, 10000, 20000]       # Test sizes for 1D algorithms
    sizes_2d = [(30, 30), (50, 50), (80, 80)]   # Test sizes for 2D DP algorithm

    # Run 1D experiments (Kadane + Divide & Conquer)
    run_experiment_1d(max_subarray_kadanes, sizes_1d, name="kadane")
    run_experiment_1d(max_subarray_sum, sizes_1d, name="dc")

    # Run 2D experiment (Max Submatrix DP)
    run_experiment_2d(sizes_2d, name="dp2d")
