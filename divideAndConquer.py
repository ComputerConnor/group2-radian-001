import numpy as np

def generate_1d_signal(n=1000, noise_level=1.0, signal_range=(10,50)):
    data = np.random.normal(0, noise_level, n)
    start = np.random.randint(0, n/2)
    length = np.random.randint(*signal_range)
    data[start:start+length] += np.random.uniform(5,10)

    return data

if __name__ == "__main__":

    data = generate_1d_signal()
    print(data)
    
    # Recursively splits arrays into two halves, finds the maximum subarray in each, combines for a final answer

# How to generate 1D synthetic datasets (used for Kadane's Greedy or Divide and Conquer algorithms):
#def generate_1d_signal(n=1000, noise_level=1.0, signal_range=(10, 50)):
#    data = np.random.normal(0, noise_level, n)
#    start = np.random.randint(0, n//2)
#    length = np.random.randint(*signal_range)
#    data[start:start+length] += np.random.uniform(5, 10)
#    return data

# How to generate Binary + Continuous Mixture arrays (used to simulate sparse genomic signals with noise, e.g. methylation)