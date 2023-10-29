import numpy as np
import argparse
import hw_1a

def Power(base, exp):
    '''
    Power Function
    return base^exp
    '''
    result = 1
    while exp:
        if exp & 1:
            result *= base
        exp >>= 1
        base *= base
    return result

def point_generator(a, w, n):
    '''
    Polynomial basis linear model data generator.
    Return a (x, y) point from y = WX + e
    '''
    error = hw_1a.gaussian_data_generator(0, a)
    x = np.random.uniform(-1, 1)
    y = error
    for i in range(n):
        y += w[i]*Power(x, i)
    return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type = int, default = 0)
    parser.add_argument('--A', type = int, default = 1)
    parser.add_argument('--W', type = str, default = '')
    args = parser.parse_args()
    n = args.N
    a = args.A
    w = []
    for k in args.W.split('/'):
        w.append(float(k))
    x, y = point_generator(a, w, n)
    print(f'The point: ({x}, {y})')