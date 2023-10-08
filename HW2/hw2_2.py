import argparse
import csv
from collections import Counter
import math
'''
python hw2-2.py --INPUT_FILE=testfile.txt --A=0 --B=0
'''

def Online_learning(count):
    '''
    Binomial_MLE = m/N
    likelihood = C{Nm} P^m (1-P)^(N-m)
    '''
    N = count[0] + count[1]
    m = count[0]
    Binomial_MLE = m / N
    return (math.factorial(N)/ (math.factorial(m) * math.factorial(N - m))) * (Binomial_MLE ** m) * ((1 - Binomial_MLE)**(N-m))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_FILE', type = str, default = 'testfile.txt')
    parser.add_argument('--A', type = int, default = 0)
    parser.add_argument('--B', type = int, default = 0)
    args = parser.parse_args()

    INPUT_FILE = args.INPUT_FILE
    a = args.A
    b = args.B

    data = []
    data_count = []

    with open(INPUT_FILE) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            data.append(row[0])
            r = list(row[0])
            count = Counter(r)
            data_count.append([int(count['1']), int(count['0'])])

    for case in range(len(data)):
        likelihood = Online_learning(data_count[case])
        # Show Result
        print(f'case{case+1}: {data[case]} ')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior: a={a}, b={b}')

        a += data_count[case][0]
        b += data_count[case][1]
        print(f'Beta posterior: a={a}, b={b}\n')