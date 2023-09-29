import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

def Power(base, exp):
    result = 1
    while exp:
        if exp & 1:
            result *= base
        exp >>= 1
        base *= base
    return result


def LSE():
    pass


if __name__ == '__main__':
    print('Start!')
    # input file, lambda and n 

    parser = argparse.ArgumentParser()
    parser.add_argument('--FILENAME', type = str, default = 'testfile.txt')
    parser.add_argument('--N', type = int, default = 3)
    parser.add_argument('--LAMBDA', type = int, default = 10000)
    args = parser.parse_args()

    INPUT_FILE = args.FILENAME
    N = args.N
    LAMBDA = args.LAMBDA

    point = []
    with open(INPUT_FILE, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            point.append([float(row[0]), float(row[1])])
    data = np.array(point)
    # print(data)

    '''
    Gram_matrix
    '''

    x = data[:,0]
    b = data[:,1].reshape((len(data[:,1]), 1))
    A = []
    for x_ in x:
        row = [Power(x_, i) for i in range(N-1, -1, -1)]
        A.append(tuple(row))
    # print(A)
    A = np.array(A)
    gram_matrix = np.matmul(A.T, A)
    # print(gram_matrix)
    AT_b = np.matmul(A.T, b)
    print(AT_b)

    '''
    LSE
    1. Use LU decomposition to find the inverse of , Gauss-Jordan elimination will also be accepted.(A is the design matrix).
    2. Print out the equation of the best fitting line and the error.
    '''

    lambda_I = LAMBDA* np.identity( np.shape(gram_matrix)[0])
    gram_matrix_add_lambda_I = np.add(gram_matrix, lambda_I)
    # print(gram_matrix_add_lambda_I)

    # LU
    