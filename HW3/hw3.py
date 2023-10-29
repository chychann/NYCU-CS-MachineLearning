import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hw_1b

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type = float, default= 1)
    parser.add_argument('--N', type = float, default = 4)
    parser.add_argument('--A', type = float, default = 1)
    parser.add_argument('--W', type = str, default = '[1,2,3,4]')
    args = parser.parse_args()
    b = args.B
    n = args.N
    a_for_point = args.A
    w = []
    str_values = args.W.strip('[]').split(',')
    w = [float(value) for value in str_values]
    k, mean, covariance = 0, 0, 0
    new_mean, new_variance = 0, 0
    data_x, data_y = [], []
    previous_mu = np.full((n), -100) # not important ar first round
    previous_covariance = np.full((n, n), -100)

    while True:
        k += 1
        x, y = hw_1b.point_generator(a_for_point, w, n)
        data_x.append(x)
        data_y.append(y)
        #design matirx = A
        A = np.zeros((1,n), dtype = float)
        for i in range(n):
            A[0][i] = Power(x, i)
        # calculate the mean and variance of data points to get 'a'
        tmp = (x - new_mean) ** 2
        new_mean = (new_mean * (k-1) + x) / k
        new_variance = ((k-1) * new_variance + tmp + (x - new_mean) ** 2) / k
        a = 1e-5 if new_variance == 0 else new_variance
        # update the prior
        if k == 1:
            sigma = np.linalg.inv( a * A.T.dot(A) + b * np.identity(n))
            mu = a * sigma.dot(A.T).dot([[y]])
        else:
            sigma = np.linalg.inv( a * A.T.dot(A) + np.linalg.inv(previous_covariance))
            #mu = sigma.dot(a * A.transpose().dot([[y]]) + np.linalg.inv(previous_covariance).dot(previous_mu))
            mu =  sigma.dot(a *(A.T).dot([[y]]) + np.linalg.inv(previous_covariance).dot(previous_mu))

        #calculate the parameters of predictive distribution
        predictive_mean = A.dot(mu)
        predictive_variance = (1/a) + A.dot(sigma).dot(A.T)
        
        # print out the result
        print(f'Add data point : ({x}, {y})')
        print('Posterior mean:')
        for i in range(n):
            print(mu[i][0])
        print()
        print('Posterior variance:')
        for i in range(n):
            for j in range(n):
                print(sigma[i][j], end = ' ')
            print()
        print()
        print(f'Predictive distribution ~ N({predictive_mean[0][0]}, {predictive_variance[0][0]})')
        print()
        print('==========')
        
        if k == 10:
            data_x_10 = data_x.copy()
            data_y_10 = data_y.copy()
            mu_10 = mu.copy()
            sigma_10 = sigma.copy()
            a_10 = a
        if k == 50:
            data_x_50 = data_x.copy()
            data_y_50 = data_y.copy()
            mu_50 = mu.copy()
            sigma_50 = sigma.copy()
            a_50 = a
        
        # check if converge
        error = 0
        for i in range(len(mu)):
            error += (previous_mu[i] - mu[i]) ** 2
        if (k > 1 and error < 1e-10) or k > 10000:
            break
        
        # if no converge, update
        mean = new_mean
        covariance = new_variance
        previous_mu = mu
        previous_covariance = sigma

    ## ground truth
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title('Ground truth')
    ground_func = np.poly1d(np.flip(w))
    ground_x = np.linspace(-2.0, 2.0, 30)
    ground_y = ground_func(ground_x)
    plt.plot(ground_x, ground_y, color = 'black')
    ground_y += a_for_point                          #mean + variance
    plt.plot(ground_x, ground_y, color = 'red')
    ground_y -= 2 * a_for_point                      #mean + variance - 2 * variance
    plt.plot(ground_x, ground_y, color = 'red')
    fig.savefig('Ground_truth.png')
    
    ## predict result
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title('Predict result')
    predict_x = np.linspace(-2.0, 2.0, 30)
    predict_func = np.poly1d(np.flip(mu.flatten()))
    predict_y = predict_func(predict_x)
    predict_y_plus = predict_func(predict_x)
    predict_y_minus = predict_func(predict_x)

    for i in range(len(predict_x)):
        A = np.zeros((1, n), dtype=float)
        for j in range(n):
            A[0][j] = (predict_x[i] ** j)
        predict_predictive_distribution_variance = 1 / a + A.dot(sigma).dot(A.T)[0][0]
        # print(predict_predictive_distribution_variance)
        predict_y_plus[i] += predict_predictive_distribution_variance
        predict_y_minus[i] -= predict_predictive_distribution_variance

    plt.plot(predict_x, predict_y, color = 'black')
    plt.plot(predict_x, predict_y_plus, color = 'red')
    plt.plot(predict_x, predict_y_minus, color = 'red')
    plt.scatter(data_x, data_y)
    fig.savefig('Predict_result.png')
    if k >= 10:
        ## After 10 incomes
        fig = plt.figure()
        plt.xlim(-2.0, 2.0)
        plt.title('After 10 incomes')
        predict_func = np.poly1d(np.flip(mu_10.flatten()))
        predict_y = predict_func(predict_x)
        predict_y_plus = predict_func(predict_x)
        predict_y_minus = predict_func(predict_x)

        for i in range(len(predict_x)):
            A = np.zeros((1, n), dtype=float)
            for j in range(n):
                A[0][j] = (predict_x[i] ** j)
            predict_predictive_distribution_variance = 1 / a_10 + A.dot(sigma_10).dot(A.T)[0][0]
            # print(predict_predictive_distribution_variance)
            predict_y_plus[i] += predict_predictive_distribution_variance
            predict_y_minus[i] -= predict_predictive_distribution_variance

        plt.plot(predict_x, predict_y, color = 'black')
        plt.plot(predict_x, predict_y_plus, color = 'red')
        plt.plot(predict_x, predict_y_minus, color = 'red')
        plt.scatter(data_x_10, data_y_10)
        fig.savefig('After_10_data_points.png')
    if k >= 50:
        ## After 50 incomes
        fig = plt.figure()
        plt.xlim(-2.0, 2.0)
        plt.title('After 50 incomes')
        predict_func = np.poly1d(np.flip(mu_50.flatten()))
        predict_y = predict_func(predict_x)
        predict_y_plus = predict_func(predict_x)
        predict_y_minus = predict_func(predict_x)

        for i in range(len(predict_x)):
            A = np.zeros((1, n), dtype=float)
            for j in range(n):
                A[0][j] = (predict_x[i] ** j)
            predict_predictive_distribution_variance = 1 / a_50 + A.dot(sigma_50).dot(A.T)[0][0]
            # print(predict_predictive_distribution_variance)
            predict_y_plus[i] += predict_predictive_distribution_variance
            predict_y_minus[i] -= predict_predictive_distribution_variance

        plt.plot(predict_x, predict_y, color = 'black')
        plt.plot(predict_x, predict_y_plus, color = 'red')
        plt.plot(predict_x, predict_y_minus, color = 'red')
        plt.scatter(data_x_50, data_y_50)
        fig.savefig('After_50_data_points.png')