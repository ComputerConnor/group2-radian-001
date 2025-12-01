import numpy as np

def generate_2d_signal(n=100, m=100, noise_level=1.0, signal_shape=(10,15)):
    dataset = np.random.normal(0, noise_level, (n,m))
    i = np.random.randint(0, n - signal_shape[0])
    j = np.random.randint(0, m - signal_shape[1])
    dataset[i:i+signal_shape[0], j:j+signal_shape[1]] += np.random.uniform(5,10)

    return dataset

def max_subarray_kadanes(arr):

    max_sum = arr[0]
    current_sum = arr[0]
    starting_index = 0
    current_index = 1
    final_index = len(arr)
    for x in arr[1:]:
        current_sum = max(x, current_sum + x)
        if current_sum == current_sum+x:
            starting_index = current_index
        max_sum = max(max_sum, current_sum)
        final_index = current_index
        current_index += 1

    print(f"Starting Index: {starting_index} - Final Index: {final_index}")

    return max_sum

def max_submatrix(mat):
    rows = len(mat)
    cols = len(mat[0])

    max_sum = float('-inf')
    temp = [0] * rows
    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += mat[row][right]
            sum_value = max_subarray_kadanes(temp)
            max_sum = max(max_sum, sum_value)

    return max_sum


if __name__ == "__main__":

    #data = generate_2d_signal()
    data = np.random.randint(-10,10,(5,5))
    print(data)

    print("\n")
    print(f"Maximum subarray: {max_submatrix(data)}")