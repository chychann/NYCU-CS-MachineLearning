import numpy as np
import argparse

def gaussian_data_generator(mean, variance):
    # Generate two uniform random numbers
    u1, u2 = np.random.uniform(0, 1, 2)

    # Box-Muller transformation
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    # Scale and shift to get the desired mean and variance
    # Since z0 and z1 are standard normals, we only need one of them
    data_point = np.sqrt(variance) * z0 + mean

    return data_point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type = int, default = 0)
    parser.add_argument('--variance', type = int, default = 1)
    args = parser.parse_args()
    mean = args.mean
    variance = args.variance
    data_point = gaussian_data_generator(mean, variance)
    print(data_point)