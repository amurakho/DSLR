import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# def plot_des(data, info):
#     """
#     WORKING ONLY WITH TWO CLASSES
#     :param info:
#         DataFrame
#         Train class
#     Plot scaterplot of two classes with desigion boundary
#     """
#     min_x = np.min(data.iloc[:, 0])
#     max_x = np.max(data.iloc[:, 0])
#
#     b = -(info.weights[2] / info.weights[1])
#     m = info.weights[0] / info.weights[1]
#     min_corr = b - m * min_x
#     max_corr = b - m * max_x
#
#     plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:, 2])
#     plt.plot([min_corr, max_corr], [min_x, max_x])
#     plt.show()


# def plot_reg(data, info):
#     """
#     WORKING ONLY WITH TWO CLASSES
#     :param info:
#         DataFrame
#         Train class
#     Plot the scatter with sigmoid func
#     Create only for test
#     """
#
#     x = np.linspace(np.min(data.iloc[:, :-1]), np.max(data.iloc[:, :-1]), 100)
#     x = pd.DataFrame(x)
#     y = info.predict_prob(x)
#     print(x)
#     print(y)
#     plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:, 1])
#     plt.scatter(x, y)
#     plt.show()


def std_data(data):
    """
    Standartization data
    """
    c_data = data.copy()
    for i in range(np.shape(data)[1] - 1):
        c_data.iloc[:, i] = (data.iloc[:, i] - np.mean(data.iloc[:, i])) / np.std(data.iloc[:, i])
    return c_data


class Logreg(object):
    """
    Base class Logistic Regression
    With Multi-Class option

    :param_data:
        Learning rate
        epoch
        classes number
        safe_w: if need to save weight into the file (weights.csv)

    Short finc description:
    predict:
        Predicts and returns the final result

    predict_prob:
        Predict result(in numbers). Good for test
        and plot Hist

    sigmid_func:
        Base sigmoid func

    make_binary:
        Make label from 'word' to binary number

    train_func:
        Base algo of Logistic Regression

    fit:
        Initialize weight, clean data from Nan
        and start training
    """

    def __init__(self,
                 learning_rate=1,
                 epoch=100,
                 classes_num=2,
                 save_w=False):
        self.learning_rate = learning_rate
        self.epoch = epoch
        if classes_num < 2:
            print("Error! Min 2")
            exit(1)
        self.classes_num = classes_num
        self.save_w = save_w

    def predict(self,
                data):
        """
        Predicts and returns the final result
        :param data:
            DataFrame base which need to predict
        :return:
            Class(Base of classes) which was predict
            (only name of class)
        Take the best the result of predict and make it to class
        """

        res = self.predict_prob(data).idxmax(axis=1)
        return self.classes_name[res]

    def predict_prob(self,
                     data):
        """
        Predict result(in numbers). Good for test
        and plot Hist
        :param data:
            DataFrame base which need to predict
        :return:
            DataFrame for all classes(only numbers)
        """

        data = data.dropna()
        predict = pd.DataFrame()
        for i in range(self.classes_num):
            print(np.dot(self.weights[i][:-1], data.T) + self.weights[i][-1])
            predict[i] = self.sigmid_func(np.dot(self.weights[i][:-1], data.T) + self.weights[i][-1])
        return predict

    def sigmid_func(self,
                    sum_res):
        """
        Base sigmoid func
        :param sum_res:
            Result of sum function
            (w0 + x1*w1 + x2*w2 + ... + xNwN)
        """

        sig_res = 1 / (1 + np.exp(-sum_res))
        return sig_res

    def make_binary(self,
                    data,
                    class_count):
        """
        Make label from 'word' to binary number
            DataFrame  base which need to make binary
            number of class with the algo working
        :param make_binary:
            DataFrame which classes need to make binary
        """

        self.classes_name = data['Hogwarts House'].unique()
        data['Hogwarts House'] = np.where(data['Hogwarts House'] == self.classes_name[class_count], 1, 0)

    def train_func(self,
                   data,
                   class_count):
        """
        Base algo of Logistic Regression
        :param class_count:
            DataFrame information on which to study
            Number of class
        :return:
            cost numpy-array
            class info

        1. Make binary
        2. take sigmoid result
        3. Take cost (1/m * (label * log(sig) + (1 - label) * log(1 - sig)))
        4. Take error and make gradient descent
        2. While epoch
        """

        self.make_binary(data, class_count)
        costs = []
        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]
        for i in range(self.epoch):
            sig_res = self.sigmid_func(np.dot(self.weights[class_count][:-1], x.T) + self.weights[class_count][-1])
            cost = -np.mean((y * np.log(sig_res)) +
                                    ((1 - y) * np.log(1 - sig_res)))

            error = sig_res - y
            grad_b = np.mean(error)
            grad_w = (1 / np.shape(x)[0]) * np.dot(error, x)

            self.weights[class_count][:-1] -= grad_w * self.learning_rate
            self.weights[class_count][-1] -= grad_b * self.learning_rate

            costs.append(cost)
            # print("An the {} iteration, sig are {}".format(i, sig_res))
        return costs, self

    def fit(self,
            data):
        """
        Initialize weight, clean data from Nan
        and start training
        :param fit:
            DataFrame on which to study
        :return:
            weight result
        Initial weight
        train for every class
        """

        self.weights = np.zeros([self.classes_num, np.shape(data)[1]])
        for i in range(self.classes_num):
            c_data = data.copy().dropna()
            self.train_func(c_data, i)
        if self.save_w:
            weight = pd.DataFrame(self.weights)
            weight.to_csv('weights.csv')
            print('It saves')




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
        data = df[['Astronomy', 'Herbology', 'Hogwarts House']]

        data = std_data(data)

        l = Logreg(classes_num=4, save_w=True)

        # # For test
        # data = data.iloc[:, ::2]
        # data = data.loc[(data['Hogwarts House'] =='Slytherin') | (data['Hogwarts House'] =='Hufflepuff')]
        # data['Hogwarts House'] = np.where(data['Hogwarts House'] == 'Hufflepuff', 1, 0)
        # # For test

        l.fit(data)

        # # test
        # # WORKING ONLY WITH TWO CLASSES
        # plot_reg(data, l)
        # # test
        ## Test

        # predict = l.predict(data)
        # tmp = data.dropna()
        # tmp = tmp.iloc[:, -1]
        # error = 0
        # for i, val in enumerate(tmp):
        #     if predict[i] != val:
        #         print("{} Predict value is '{}' but base value is  '{}'".format(i, predict[i], val))
        #         error +=1
        # print("Number of errors:", error)
