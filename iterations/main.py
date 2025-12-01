import numpy as np
import time

# SYNTHETIC DATASET GENERATION

# This algorithm generates an array of
def generate_1d_signal(n: int, noise_level=1.0, signal_range=(10, 50)):
    data = np.random.normal(0, noise_level, n)
    start = np.random.randint(0, n / 2)
    length = np.random.randint(*signal_range)
    data[start:start + length] += np.random.uniform(5, 10)

    return data

def generate_2d_signal(n: int, m: int, noise_level=1.0, signal_shape=(10,15)):
    dataset = np.random.normal(0, noise_level, (n,m))
    i = np.random.randint(0, n - signal_shape[0])
    j = np.random.randint(0, m - signal_shape[1])
    dataset[i:i+signal_shape[0], j:j+signal_shape[1]] += np.random.uniform(5,10)

    return dataset

# DIVIDE AND CONQUER
def max_crossing_sum(array, l, m, h):

    # Include elements on the left of mid
    left_total = float('-inf')
    total = 0
    for i in range(m, l - 1, -1):
        total += array[i]
        left_total = max(left_total, total)

    # Include elements on the right of mid
    right_total = float('-inf')
    total = 0
    for i in range(m, h + 1):
        total += array[i]
        right_total = max(right_total, total)

    # Return the maximum of left sum, right sum, and their combination
    return max(left_total + right_total - array[m], left_total, right_total)


def max_subarray_dc(array, l, h):
    if l > h:
        return float('-inf')
    if l == h:

        # Base case: one element
        return array[l]

    # Find the middle point
    m = l + (h - l) // 2

    # Return the maximum of three cases
    return max(
        max_subarray_dc(array, l, m),
        max_subarray_dc(array, m + 1, h),
        max_crossing_sum(array, l, m, h),
    )

def max_subarray_sum(array):
    return max_subarray_dc(array, 0, len(array) - 1)

# GREEDY AND KADANE'S ALGORITHM
def max_subarray_kadanes(array):

    max_sum = array[0]
    current_sum = array[0]
    for x in array[1:]:
        current_sum = max(x, current_sum + x)
        max_sum = max(max_sum, current_sum)

    return max_sum

# 2D EXPANSION WITH DP
def max_submatrix(mat):
    rows = len(mat)
    cols = len(mat[0])

    max_sum = float('-inf')
    #temp = [0] * rows
    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += mat[row][right]
            sum_value = max_subarray_kadanes(temp)
            max_sum = max(max_sum, sum_value)

    return max_sum

if __name__ == "__main__":

    # Divide and Conquer
    data1 = generate_1d_signal(n = 1000)
    data2 = generate_1d_signal(n = 10000)
    data3 = generate_1d_signal(n = 100000)

    print("\nDIVIDE AND CONQUER TESTING\n")

    print("TEST CASE: [2, 3, -8, 7, -1, 2, 3] - Maximum subarray is 11: [7,-1,2,3]")

    testdata = [2, 3, -8, 7, -1, 2, 3]
    print(f"Maximum subarray: {max_subarray_sum(testdata)}\n")

    print("TESTS WITH 1D SYNTHETIC DATASETS:\n")
    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_sum(data1)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_sum(data2)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_sum(data3)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")

    # Greedy and Kadane's Algorithm
    print("GREEDY/KADANE'S TESTING\n")

    print("TEST CASE: [2, 3, -8, 7, -1, 2, 3] - Maximum subarray is 11: [7,-1,2,3]")
    print(f"Maximum subarray: {max_subarray_kadanes(testdata)}\n")

    print("TESTS WITH 1D SYNTHETIC DATASETS (SAME DATASETS AS DIVIDE AND CONQUER TESTING):\n")
    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_kadanes(data1)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 6)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_kadanes(data2)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 6)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_subarray_kadanes(data3)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 6)}\n")

    # 2D Expansion with Dynamic Programming

    print("2D EXPANSION WITH DYNAMIC PROGRAMMING TESTING\n")

    print("TEST CASE: [[1, 2, -1, -4, -20], [-8, -3, 4, 2, 1], [3, 8, 10, 1, 3], [-4, -1, 1, 7, -6]] - Maximum subarray is 29: [[-3, 4, 2], [8, 10, 1], [-1, 1, 7]]")
    testdata = mat = [[1, 2, -1, -4, -20], [-8, -3, 4, 2, 1], [3, 8, 10, 1, 3], [-4, -1, 1, 7, -6]]
    print(f"Maximum subarray: {max_submatrix(testdata)}\n")

    print("TESTS WITH 2D SYNTHETIC DATASETS:\n")
    data1 = generate_2d_signal(n = 30, m = 30)
    data2 = generate_2d_signal(n = 100, m = 100)
    data3 = generate_2d_signal(n = 150, m = 150)

    start_time = time.time()
    print(f"Maximum subarray: {max_submatrix(data1)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_submatrix(data2)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")

    start_time = time.time()
    print(f"Maximum subarray: {max_submatrix(data3)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for operation: {round(elapsed_time, 3)}\n")
