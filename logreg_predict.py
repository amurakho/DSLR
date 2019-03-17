import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys


def std_data(data):
    c_data = data.copy()
    for i in range(np.shape(data)[1] - 1):
        c_data.iloc[:, i] = (data.iloc[:, i] - np.mean(data.iloc[:, i])) / np.std(data.iloc[:, i])
    return c_data

def sigmid_func(sum_res):
    sig_res = 1 / (1 + np.exp(-sum_res))
    return sig_res


def predict_prob(data, weights):
    predict = pd.DataFrame()
    weights = weights.iloc[:, 1:]
    for i, weight in weights.iterrows():
        predict[i] = sigmid_func(np.dot(weight[:-1], data.T) + np.round(weight[-1], 6))
    return predict

def predict(data):
    classes_name = np.array(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
    weight = np.nan
    try:
        weight = pd.read_csv('weights.csv')
    except:
        print("Should train model")
        exit(1)
    res = predict_prob(data, weight).idxmax(axis=1)
    return classes_name[res.values]



if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit(1)
    file_name = sys.argv[1]
    try:
        df = pd.read_csv(file_name)
    except:
        print("Error (file name)")
        exit(1)
    else:
        data = df[['Astronomy', 'Herbology']].dropna()
        data = std_data(data)
        my_res = predict(data)


        # Check me
        train_data = pd.read_csv('resources-3/dataset_train.csv')
        train_data = train_data[['Astronomy', 'Herbology', 'Hogwarts House']]
        train_data = std_data(train_data)

        test_data = data
        test_data = std_data(test_data)
        l = LogisticRegression(solver='lbfgs', multi_class='multinomial')

        train_data = train_data.dropna()
        test_data = test_data.dropna()
        x = train_data.iloc[:, :-1]
        y = train_data.iloc[:, -1]

        l.fit(x, y)
        base_res = l.predict(test_data)
        error = 0
        for i, val in enumerate(my_res):
            if my_res[i] != base_res[i]:
                error += 1
                print("{} My result is {}, and base result is {}".format(i, my_res[i], base_res[i]))
        print("Full error: ", error)
