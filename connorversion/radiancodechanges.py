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
# ============================================================

def generate_1d_signal(n: int, noise_level=1.0, signal_range=(10, 50)):
    data = np.random.normal(0, noise_level, n)
    start = np.random.randint(0, n // 2)
    length = np.random.randint(*signal_range)
    data[start:start + length] += np.random.uniform(5, 10)
    return data


def generate_2d_signal(n: int, m: int, noise_level=1.0, signal_shape=(10, 15)):
    dataset = np.random.normal(0, noise_level, (n, m))
    i = np.random.randint(0, n - signal_shape[0])
    j = np.random.randint(0, m - signal_shape[1])
    dataset[i:i + signal_shape[0], j:j + signal_shape[1]] += np.random.uniform(5, 10)
    return dataset


# ============================================================
# DIVIDE AND CONQUER MAX SUBARRAY
# ============================================================

def max_crossing_sum(array, l, m, h):
    left_sum = float('-inf')
    total = 0
    for i in range(m, l - 1, -1):
        total += array[i]
        left_sum = max(left_sum, total)

    right_sum = float('-inf')
    total = 0
    for i in range(m + 1, h + 1):
        total += array[i]
        right_sum = max(right_sum, total)

    return left_sum + right_sum


def max_subarray_dc(array, l, h):
    if l == h:
        return array[l]
    m = (l + h) // 2
    return max(
        max_subarray_dc(array, l, m),
        max_subarray_dc(array, m + 1, h),
        max_crossing_sum(array, l, m, h)
    )


def max_subarray_sum(array):
    return max_subarray_dc(array, 0, len(array) - 1)


# ============================================================
# KADANE'S ALGORITHM
# ============================================================

def max_subarray_kadanes(array):
    max_sum = array[0]
    current_sum = array[0]
    for x in array[1:]:
        current_sum = max(x, current_sum + x)
        max_sum = max(max_sum, current_sum)
    return max_sum


# ============================================================
# 2D DP MAX SUBMATRIX
# ============================================================

def max_submatrix(mat):
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
# ============================================================

def run_experiment_1d(algorithm, sizes, repeats=10, name="algorithm"):
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
# ============================================================

def save_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InputSize", "AvgTime"])
        writer.writerows(rows)


def save_plot(filename, results, title="Runtime"):
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
# ============================================================

if __name__ == "__main__":
    sizes_1d = [1000, 5000, 10000, 20000]
    sizes_2d = [(30, 30), (50, 50), (80, 80)]

    # Run 1D experiments
    run_experiment_1d(max_subarray_kadanes, sizes_1d, name="kadane")
    run_experiment_1d(max_subarray_sum, sizes_1d, name="dc")

    # Run 2D experiment
    run_experiment_2d(sizes_2d, name="dp2d")
