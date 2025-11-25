import numpy as np

if __name__ = "__main__"
# Finds the maximum subarray in a 2d matrix. Fixes top and bottom rows, collapse into 1d, and apply Kadane's algorithm

# Example from radian's feedback:
#def max_submatrix(matrix):
#    n = len(matrix)
#    max_sum = float('-inf')
#    for top in range(n):
#        temp = [0]*n
#        for bottom in range(top, n):
#            for col in range(n):
#                temp[col] += matrix[bottom][col]
#            max_sum = max(max_sum, max_subarray(temp))
#    return max_sum

# How to generate 2D synthetic matrices:
#def generate_2d_signal(n=100, m=100, noise_level=1.0, signal_shape=(10, 15)):
#    data = np.random.normal(0, noise_level, (n, m))
#    i = np.random.randint(0, n - signal_shape[0])
#    j = np.random.randint(0, m - signal_shape[1])
#    data[i:i+signal_shape[0], j:j+signal_shape[1]] += np.random.uniform(5, 10)
#    return data