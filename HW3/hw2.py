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
    parser.add_argument('--mean', type = float, default = 0)
    parser.add_argument('--variance', type = float, default = 1)
    args = parser.parse_args()
    initial_mean = float(args.mean)
    initial_variance = float(args.variance)
    print(f'Data point source function: N({initial_mean}, {initial_variance})\n')

    sum_sqr = 0
    Sum = 0
    n = 0
    mean = 0
    variance = 0
    while True:
        data_point = gaussian_data_generator(initial_mean, initial_variance)
        print(f'Add data point: {data_point}')
        Sum += data_point
        sum_sqr += data_point**2
        n += 1
        m = Sum/n
        s = (sum_sqr - (Sum*Sum)/n)/(n-1) if n!=1 else 0
        print(f'Mean = {m}, Variance = {s}')
        if abs(mean - m) < 1e-4 and abs(variance - s) < 1e-4:
            break
        else:
            mean, variance = m, s