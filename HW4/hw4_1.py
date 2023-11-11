import argparse
import numpy as np
import math
import matplotlib.pyplot as plt 

'''
python hw4_1.py --N=50 --MX1=1 --MY1=1 --MX2=10 --MY2=10 --VX1=2 --VY1=2 --VX2=2 --VY2=2
python hw4_1.py --N=50 --MX1=1 --MY1=1 --MX2=3 --MY2=3 --VX1=2 --VY1=2 --VX2=4 --VY2=4
'''
def confusion_matrix(predict, y):
    ## Counfusion matrix
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for row in range(np.shape(predict)[0]):
        if predict[row][0] == 0:
            if y[row][0] == 1:
                FN += 1
            elif y[row][0] == 0:
                TN += 1
        elif predict[row][0] == 1:
            if y[row][0] == 1:
                TP += 1
            elif y[row][0] == 0:
                FP += 1

    print('\nConfusion matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print('Real cluster 1\t\t{}\t\t\t{}'.format(TN,FP))
    print('Real cluster 2\t\t{}\t\t\t{}'.format(FN,TP))
    print('\nSensitivity (Successfully predict cluster 1): {}'.format(TN / (TN + FP) ))
    print('Specificity (Successfully predict cluster 2): {}\n'.format(TP / (TP + FN) ))

def gradient_descent(A, w, learning_rate, epochs):
    for _ in range(epochs):
        exp = np.matmul(A, w)
        result = []
        for i in range(np.shape(exp)[0]):
            result.append(1/(1+np.exp((-1) * exp[i])))
        result = np.array(result)
        gradient = np.matmul(A.T, np.subtract(y, result))
        w = w + learning_rate * gradient
        if np.sqrt(np.sum(gradient**2))<1e-2:
            break
    prediction = np.matmul(A, w)
    print('Gradient descent:\n\nw:')
    for i in range(np.shape(w)[0]):
        print(w[i][0])
    return w, prediction

def Newton(A, w, learning_rate, epochs):
    for _ in range(epochs):
        exp = np.matmul(A, w)
        result = []
        for i in range(np.shape(exp)[0]):
            result.append(1/(1+np.exp((-1) * exp[i])))
        result = np.array(result)
        gradient = np.matmul(A.T, np.subtract(y, result))

        ## D
        D = np.identity(np.shape(A)[0])
        for row in range(np.shape(D)[0]):
            col = row
            fraction = np.exp((-1) * np.matmul(A[row][:], w))
            if math.isinf(fraction):
                fraction = np.exp(700)
            D[row][col] =  fraction / ((1 + fraction) ** 2)
            if math.isnan(D[row][col]):
                D[row][col] = np.random.random_sample() * 100
        hessian = np.matmul(np.matmul(A.T, D), A)
        if np.linalg.matrix_rank(hessian) == hessian.shape[0]:

            w = w + learning_rate * np.matmul(np.linalg.inv(hessian), gradient)
        else:
            w = w + learning_rate * gradient

        if np.sqrt(np.sum(gradient**2))<1e-2:
            break
    prediction = np.matmul(A, w)
    print('Newton\'s:\n\nw:')
    for i in range(np.shape(w)[0]):
        print(w[i][0])
    return w, prediction

def gaussian_data_generator(mean, variance):
    u1, u2 = np.random.uniform(0, 1, 2)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    data_point = np.sqrt(variance) * z0 + mean
    return data_point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type = int, default = 50)
    parser.add_argument('--MX1', type = float, default = 1)
    parser.add_argument('--VX1', type = float, default = 2)
    parser.add_argument('--MY1', type = float, default = 1)
    parser.add_argument('--VY1', type = float, default = 2)

    parser.add_argument('--MX2', type = float, default = 10)
    parser.add_argument('--VX2', type = float, default = 2)
    parser.add_argument('--MY2', type = float, default = 10)
    parser.add_argument('--VY2', type = float, default = 2)
    args = parser.parse_args()
    N = args.N
    MX1 = args.MX1
    VX1 = args.VX1
    MY1 = args.MY1
    VY1 = args.VY1
    MX2 = args.MX2
    VX2 = args.VX2
    MY2 = args.MY2
    VY2 = args.VY2

    epochs = 1000
    learning_rate = 0.15

    Data_1, Data_2 = [], []
    Data_y_1, Data_y_2 = [], []
    y = []
    design_matrix = []

    for i in range(N):
        dx1 = gaussian_data_generator(MX1, VX1)
        dy1 = gaussian_data_generator(MY1, VY2)
        dx2 = gaussian_data_generator(MX2, VX2)
        dy2 = gaussian_data_generator(MY2, VY2)

        Data_1.append([dx1, dy1])
        y.append(0)
        Data_2.append([dx2, dy2])
        y.append(1)
        design_matrix.append([1, dx1, dy1])
        design_matrix.append([1, dx2, dy2])
    design_matrix = np.array(design_matrix)
    Data_1 = np.array(Data_1)
    Data_2 = np.array(Data_2)
    y = np.array(y).reshape(-1, 1)
    w = np.random.randn(design_matrix.shape[1],1).reshape(-1,1)
    weight_grad, pred_grad = gradient_descent(design_matrix, w, learning_rate, epochs)
    grad_1, grad_2 = [], []
    for row in range(np.shape(pred_grad)[0]):
        if pred_grad[row] > 0.5:
            grad_1.append([design_matrix[row][1], design_matrix[row][2]])
        else:
            grad_2.append([design_matrix[row][1], design_matrix[row][2]])
    grad_1 = np.array(grad_1)
    grad_2 = np.array(grad_2)
    pred_grad[pred_grad > 0.5] = int(1)
    pred_grad[pred_grad <= 0.5] = int(0)
    confusion_matrix(pred_grad, y)

    print('-'*20)

    w = np.random.randn(design_matrix.shape[1],1).reshape(-1,1)
    weight_newton, pred_newton = Newton(design_matrix, w, learning_rate, epochs)
    newton_1, newton_2 = [], []
    for row in range(np.shape(pred_newton)[0]):
        if pred_newton[row] > 0.5:
            newton_1.append([design_matrix[row][1], design_matrix[row][2]])
        else:
            newton_2.append([design_matrix[row][1], design_matrix[row][2]])
    newton_1 = np.array(newton_1)
    newton_2 = np.array(newton_2)
    pred_newton[pred_newton > 0.5] = int(1)
    pred_newton[pred_newton <= 0.5] = int(0)
    confusion_matrix(pred_newton, y)

    ## Plot result
    fig = plt.figure()
    plt.title('Ground Truth')
    plt.scatter(Data_1[:,0], Data_1[:,1], c = 'b')
    plt.scatter(Data_2[:,0], Data_2[:,1], c = 'r')
    fig.savefig('ground truth.png')

    fig = plt.figure()
    plt.title('Gradient Descent')
    plt.scatter(grad_1[:,0], grad_1[:,1], c = 'r')
    plt.scatter(grad_2[:,0], grad_2[:,1], c = 'b')
    fig.savefig('gradient descent.png')

    fig = plt.figure()
    plt.title("Newton's method")
    plt.scatter(newton_1[:,0], newton_1[:,1], c = 'r')
    plt.scatter(newton_2[:,0], newton_2[:,1], c = 'b')
    fig.savefig("newton's method.png")