import numpy as np

def generate_1d_signal(n=1000, noise_level=1.0, signal_range=(10, 50)):
    dataset = np.random.normal(0, noise_level, n)
    start = np.random.randint(0, n / 2)
    length = np.random.randint(*signal_range)
    dataset[start:start + length] += np.random.uniform(5, 10)

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

if __name__ == "__main__":

    #data = generate_1d_signal()
    data = np.random.randint(-10,10,10)
    print(data)

    print("\n")
    print(f"Maximum subarray: {max_subarray_kadanes(data)}")

# TODO: FIGURE OUT INDEXING LATER
"""
def max_subarray_kadanes(arr):

    start_o = 0
    final_index = 0
    starting_index = 0
    current_index = 0
    current_max = 0
    previous_max = arr[0]

    for x in arr:
        current_max += x
        if current_max < 0:
            starting_index = current_index+1
            current_max = 0
        elif current_max > previous_max:
            final_index = current_index
            start_o = starting_index
            previous_max = current_max
        current_index += 1

    print(f"Starting Index: {start_o} - Final Index: {final_index}")
"""