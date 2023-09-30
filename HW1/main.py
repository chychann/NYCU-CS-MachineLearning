import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision = 3, suppress=True) #set array format

''' 
python main.py --INPUT_FILE=testfile.txt --N=3 --LAMBDA=10000 --LR 0.0001
''' 

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

def LUdecompose(matrix):
    '''
    matrix = L * U
    
    '''
    # original setting
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1  # Diagonal of L matrix is 1

        for j in range(i, n):
            #right of the diagonal
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i] #normalize

    return L, U


def plot(model_name, x, y, b):
    '''
    Visualize the result
    x: x points
    y: real value
    b: predict value
    '''
    fig = plt.figure()
    plt.title(model_name)
    plt.plot(x, y, 'ro') #use red circle points to represent real points
    plt.plot(x, b, '-k') #use black line to represent the fitting line
    plt.show()
    fig.savefig(model_name + '.png')

def show_result(A, x, y, x1, model_name):
    '''
    print the result
    A: data, x: parameter, b: target
    b = Ax
    '''
    b = np.matmul(A,x)
    print(model_name + ': ')
    print('Fitting line : ', end='')

    for row in range(np.shape(x)[0]):
        if row != np.shape(x)[0] - 1:
            if row != 0:
                if x[row] >= 0:
                    print('+ ' + str(x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print('- ' + str((-1) * x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
            else:
                if x[row] >= 0:
                    print(str(x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print('- ' + str((-1) * x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
        else: #last
            if x[row] >= 0:
                print('+ ' + str(x[row][0]))
            else:
                print('- ' + str((-1) * x[row][0]))

    # Calculate total error
    total_error = 0
    for i in range(np.shape(b)[0]):
        total_error += np.square(b[i] - y[i])
    print('Total error: ', total_error[0])

    plot(model_name, x1, y, b)

def gradient_descent(X, y, coeffs, learning_rate, epochs, Lambda, N):
    '''
    Update the coefficients
    '''
    for _ in range(epochs):
        #Compute the current predicted value of y
        y_pred = np.dot( X, coeffs)
        # Compute the gradient of the loss with respect to coefficients
        gradient = (-2 / np.shape(X)[0]) * np.dot(X.T, (y - y_pred))
        # Add L1 regularization term to the gradients
        gradient += Lambda * np.sign(coeffs)
        # Update the coefficients using steepest descent
        coeffs -= (learning_rate * gradient)

    return coeffs


def inverse_matrix(M):
    '''
    Inverse Matrix: performs the Gauss-Jordan elimination to compute the inverse of the input matrix
    '''
    n = len(M)
    # Create an augmented matrix [M | I], where I is the identity matrix
    augmented_matrix = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            augmented_matrix[i][j] = M[i][j]
            augmented_matrix[i][j + n] = 1.0 if i == j else 0.0
    # Perform Gaussian elimination to transform [M | I] into [I | M^-1]
    for col in range(n):
        # Find the pivot row
        pivot_row = col
        for i in range(col + 1, n):
            if abs(augmented_matrix[i][col]) > abs(augmented_matrix[pivot_row][col]):
                pivot_row = i

        # Swap rows
        augmented_matrix[col], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[col]
        
        # Scale the pivot row
        pivot_val = augmented_matrix[col][col]
        for j in range(2 * n):
            augmented_matrix[col][j] /= pivot_val
        
        # Eliminate other rows
        for i in range(n):
            if i != col:
                factor = augmented_matrix[i][col]
                for j in range(2 * n):
                    augmented_matrix[i][j] -= factor * augmented_matrix[col][j]
    
    # Extract the inverse matrix [I | M^-1]
    inverse = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inverse[i][j] = augmented_matrix[i][j + n]
    
    return np.array(inverse)

if __name__ == '__main__':
    print('Start!')
    # input file, lambda and n 

    parser = argparse.ArgumentParser()
    parser.add_argument('--FILENAME', type = str, default = 'testfile.txt')
    parser.add_argument('--N', type = int, default = 3)
    parser.add_argument('--LAMBDA', type = int, default = 10000)
    parser.add_argument('--LR', type = float, default = 0.00001)
    args = parser.parse_args()

    INPUT_FILE = args.FILENAME
    N = args.N
    LAMBDA = args.LAMBDA
    LEARNING_RATE = args.LR

    point = []
    with open(INPUT_FILE, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            point.append([float(row[0]), float(row[1])])
    data = np.array(point)

    '''
    Gram_matrix
    '''

    x = data[:,0]
    b = data[:,1].reshape((len(data[:,1]), 1))
    A = []
    for x_ in x:
        row = [Power(x_, i) for i in range(N-1, -1, -1)]
        A.append(tuple(row))
    A = np.array(A)
    gram_matrix = np.matmul(A.T, A)
    AT_b = np.matmul(A.T, b)

    '''
    LSE
    1. Use LU decomposition to find the inverse of , Gauss-Jordan elimination will also be accepted.(A is the design matrix).
    2. Print out the equation of the best fitting line and the error.
    '''

    lambda_I = LAMBDA* np.identity( np.shape(gram_matrix)[0])
    gram_matrix_add_lambda_I = np.add(gram_matrix, lambda_I)
    # LU
    L, U = LUdecompose(gram_matrix_add_lambda_I)
    inverse_L = inverse_matrix(L)
    inverse_U = inverse_matrix(U)
    inverse_ATA_lambdaI = np.matmul(inverse_U, inverse_L, dtype=np.float64)
    result_lse = np.matmul(inverse_ATA_lambdaI, AT_b, dtype=np.float64)
    show_result(A, result_lse, b, x, 'LSE')
    print('\n', end='')

    '''
    steepest descent method
    1. Use steepest descent with LSE and L1 norm to find the best fitting line.
    2. Print out the equation of the best fitting line and the error.
    '''
    # Construct the design matrix
    EPOCHS = 10000
    coefficients = np.zeros((N,1))  # Initialize coefficients
    # Performing Gradient Descent 
    coefficients = gradient_descent(A, b, coefficients, LEARNING_RATE, EPOCHS, LAMBDA, N)
    show_result(A, coefficients, b, x, 'Descent')

    ''' 
    Newton's method
    Xn+1 = Xn - [H]^-1 * gradient
    Gradient = 2 * A_transpose * A * x - 2 * A_transpose * b
    Hession = 2 * A_transpose * A
    '''
    result = np.full((N, 1), 100)
    AT_A_x = np.matmul(gram_matrix,result)
    gradient = 2 * (AT_A_x - AT_b)
    hessian = 2 * (np.matmul(A.T, A))

    L, U = LUdecompose(hessian)
    inverse_L_N = inverse_matrix(L)
    inverse_U_N = inverse_matrix(U)
    inverse_hessian = inverse_U_N.dot(inverse_L_N)
    # print(inverse_hessian)
    update = np.matmul(inverse_hessian, gradient)
    result = result - update
    show_result(A, result, b, x, 'Newton\'s')