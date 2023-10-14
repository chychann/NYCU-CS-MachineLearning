import argparse
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def train_openfile(img_path, label_path):
    file_img = open(img_path, 'rb')
    file_label = open(label_path, 'rb')
    # img
    global train_image_magic, train_image_num_img, train_image_row, train_image_col, train_label_magic, train_label_item
    train_image_magic = int.from_bytes(file_img.read(4), byteorder='big')
    train_image_num_img = int.from_bytes(file_img.read(4), byteorder='big')
    train_image_row = int.from_bytes(file_img.read(4), byteorder='big')
    train_image_col = int.from_bytes(file_img.read(4), byteorder='big')
    global train_num_pixels
    train_num_pixels = train_image_row * train_image_col
    # label
    train_label_magic = int.from_bytes(file_label.read(4), byteorder='big')
    train_label_item = int.from_bytes(file_label.read(4), byteorder='big')
    return file_img, file_label

def test_openfile(img_path, label_path):
    file_img = open(img_path, 'rb')
    file_label = open(label_path, 'rb')
    # img
    global test_image_magic, test_image_num_img, test_image_row, test_image_col, test_label_magic, test_label_item
    test_image_magic = int.from_bytes(file_img.read(4), byteorder='big')
    test_image_num_img = int.from_bytes(file_img.read(4), byteorder='big')
    test_image_row = int.from_bytes(file_img.read(4), byteorder='big')
    test_image_col = int.from_bytes(file_img.read(4), byteorder='big')
    global test_num_pixels
    test_num_pixels = test_image_row * test_image_col
    # label
    test_label_magic = int.from_bytes(file_label.read(4), byteorder='big')
    test_label_item = int.from_bytes(file_label.read(4), byteorder='big')
    return file_img, file_label

def Normalize(prob):
    '''
    Normalize the probabilities so that they sum up to 1.
    '''
    return prob / np.sum(prob)

def Result(prob, ans):
    '''
    Display the posterior probabilities in log scale and the prediction.
    '''
    print('Posterior (in log scale):')
    prediction = np.argmin(prob)
    for i in range(label_num):
        print(f'{i}: {prob[i]}')
    print(f'Prediction: {prediction}, Ans: {ans}')
    # Return 0 if the prediction is correct and 1 otherwise.
    return 0 if prediction == ans else 1

def print_imagination_discrete(likelihood):
    print('Imagination of numbers in Bayesian classifier:', '\n')
    for i in range(label_num):
        print(i, ':')
        for m in range(train_image_row * train_image_col):  # 28 * 28 = 784
            # The pixel is 0 when Bayes classifier expects the pixel in this position should be less than 128 in the original image, otherwise, it's 1.
            temp = 0
            for t in range(16):  # bin 0~15
                temp += likelihood[i][m][t]
            for t in range(16, 32):  # bin 16~32
                temp -= likelihood[i][m][t]
            if temp > 0:
                print('0', end=' ')
            else:
                print('1', end=' ')

            # To print a newline at the end of every 28 pixels (equivalent to the original inner loop)
            if (m + 1) % 28 == 0:
                print('\n')
    print('\n')

def print_imagination_continuous(mean):
    print('Imagination of numbers in Bayesian classifier:', '\n')
    for i in range(label_num):
        print(i, ':')
        for m in range(train_image_row * train_image_col):  # 28 * 28 = 784
            # The pixel is 0 when Bayes classifier expects the pixel in this position should be less than 128 in the original image, otherwise, it's 1.
            if mean[i][m] < 128:
                print('0', end=' ')
            else:
                print('1', end=' ')

            # To print a newline at the end of every 28 pixels (equivalent to the original inner loop)
            if (m + 1) % 28 == 0:
                print('\n')
    print('\n')

def log_gaussian_pdf(x, mean, variance):
    """
    Compute the log of the Gaussian probability density function for a given x, mean, and variance.
    """
    return np.log(1/ np.sqrt(2 * np.pi * variance)) - (((x - mean) ** 2) / (2 * variance))

def discrete_mode():
    prior = np.zeros(label_num, dtype=int)
    likelihood = np.zeros((label_num, train_image_row * train_image_col, 32), dtype = int)
    for i in range(train_image_num_img):
        label = int.from_bytes(file_train_label.read(1), byteorder='big')
        prior[label]+=1
        # Tally the frequency of the values of each pixel into 32 bins
        for pix_id in range(train_num_pixels):
            pixel = int.from_bytes(file_train_image.read(1), byteorder='big')
            likelihood[label][pix_id][int(pixel/8)] += 1
            # 1 bytes = 8 bits = 256 ; 256/8 = 32

    likelihood_sum = np.zeros((label_num, train_image_row * train_image_col), dtype=int)
    for i in range(label_num):
        likelihood_sum[i] = np.sum(likelihood[i], axis=1)

    # Testing
    error = 0
    for i in range(test_image_num_img):#
        test_imgs = np.zeros((test_image_row * test_image_col), dtype=int)
        for j in range(test_image_row * test_image_col):
            # calculate the bins of pixel
            test_imgs[j] = int(int.from_bytes(file_test_image.read(1), byteorder='big') / 8)
        prob = np.zeros((label_num), dtype=float)
        answer = int.from_bytes(file_test_label.read(1), byteorder='big')
        for j in range(label_num):
            prob[j] = np.log(float(prior[j] / train_image_num_img))
            for k in range(test_image_row * test_image_col):
                val = likelihood[j][k][test_imgs[k]]
                if val:
                    prob[j] += np.log( float(val) / likelihood_sum[j][k])
                else:
                    prob[j] += np.log(float(1e-6 / likelihood_sum[j][k]))
        prob = Normalize(prob)
        error += Result(prob, answer)

    # close all file
    file_train_image.close()
    file_train_label.close()
    file_test_image.close()
    file_test_label.close()
    #print imagination
    print_imagination_discrete(likelihood)
    print(f'Error rate: {float(error / test_image_num_img )}')

def continuous_mode():
    '''
    1. Use MLE to fit a Gaussian distribution for the value of each pixel.
    2. Perform Naive Bayes classifier.
    '''
    prior = np.zeros((label_num), dtype=float)
    pixel_mean = np.zeros((label_num, train_image_row * train_image_col), dtype=float)
    pixel_var = np.zeros((label_num, train_image_row * train_image_col), dtype=float)
    pixel_square = np.zeros((label_num, train_image_row * train_image_col), dtype=float)

    for _ in range(train_image_num_img):
        label = int.from_bytes(file_train_label.read(1), byteorder='big')
        prior[label] += 1
        for pixel_id in range(train_num_pixels):
            pixel = int.from_bytes(file_train_image.read(1), byteorder='big')
            pixel_square[label][pixel_id] += float(pixel**2)
            pixel_mean[label][pixel_id] += float(pixel)
    for label in range(label_num):
        for pixel_id in range(train_num_pixels):
            pixel_mean[label][pixel_id] = float(pixel_mean[label][pixel_id] / prior[label])
            pixel_var[label][pixel_id] = float(pixel_square[label][pixel_id] / prior[label])  - float(pixel_mean[label][pixel_id] ** 2)

            if pixel_var[label][pixel_id] == 0:
                pixel_var[label][pixel_id] = 1e-4

    # log probabilities 
    prior = prior / train_image_num_img
    prior = np.log(prior)
    print(prior)

    # Test
    error = 0
    for _ in range(test_image_num_img):
        test_img = np.zeros((test_image_row * test_image_col), dtype=float)
        for pixel_id in range(test_image_row * test_image_col):
            # calculate the bins of pixel
            test_img[pixel_id] = int.from_bytes(file_test_image.read(1), byteorder='big')
        answer = int.from_bytes(file_test_label.read(1), byteorder='big')
        prob = np.zeros((label_num), dtype=float)
        for j in range(label_num):
            prob[j] += prior[j]
            for pixel_id in range(test_image_row * test_image_col):
                test_prob = log_gaussian_pdf(test_img[pixel_id], pixel_mean[j][pixel_id], pixel_var[j][pixel_id])
                prob[j] += test_prob
        prob = Normalize(prob)
        error += Result(prob, answer)
    # close all file
    file_train_image.close()
    file_train_label.close()
    file_test_image.close()
    file_test_label.close()

    #print imagination
    print_imagination_continuous(pixel_mean)
    print(f'Error rate: {float(error / test_image_num_img )}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAINING_IMAGE', type = str, default = 'train-images-idx3-ubyte')
    parser.add_argument('--TRAINING_LABEL', type = str, default = 'train-labels-idx1-ubyte')
    parser.add_argument('--TESTING_IMAGE', type = str, default = 'tlabel_numk-images-idx3-ubyte')
    parser.add_argument('--TESTING_LABEL', type = str, default = 'tlabel_numk-labels-idx1-ubyte')
    parser.add_argument('--OPTION', type = int, default = 0)
    args = parser.parse_args()

    training_image_path = args.TRAINING_IMAGE
    training_label_path = args.TRAINING_LABEL
    testing_image_path = args.TESTING_IMAGE
    testing_label_path = args.TESTING_LABEL
    toggle_mode = args.OPTION

    global file_train_image, file_train_label, file_test_image, file_test_label
    file_train_image, file_train_label = train_openfile(training_image_path, training_label_path)
    file_test_image, file_test_label = test_openfile(testing_image_path, testing_label_path)

    global label_num
    label_num = 10

    if toggle_mode == 0:
        discrete_mode()
    else:
        continuous_mode()