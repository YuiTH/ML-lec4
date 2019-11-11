import numpy as np
import readFile
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt


class softmax_regression():
    def __init__(self,num_feature=2,num_class=3,num_data=150,learning_rate=1.5):
        self.num_feature = num_feature
        self.num_class = num_class
        self.num_data = num_data
        self.learning_rate = learning_rate
        self.weight = np.random.randn(self.num_feature, self.num_class)
        self.bias = np.ones(self.num_data, )
        self.color_list = ['r', 'g', 'b', 'y']


    def normalize(self, data):
        data = data - np.mean(data, axis=0, keepdims=True)
        data = data / (np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
        return data

    def softmax(self, X):  # softmax函数
        return np.exp(X) / np.sum(np.exp(X))


    def loss(self,X,label):
        score = np.dot(X, self.weight)
        score -= np.max(score, axis=1, keepdims=True)
        sum_exp_score = np.sum(np.exp(score), axis=1)
        loss = np.log(sum_exp_score)
        # remove redundant term: the correct term
        loss -= score[np.arange(self.num_data), label]
        loss = (1. / self.num_data) * np.sum(loss)
        return loss

    def update_parameter(self, ):
        ''' compute grediant. '''
        rand_i = np.random.randint(0, self.num_data, stochastic)
        x = self.data[rand_i]
        y = self.label[rand_i]
        softmax = self.softmax(x)
        softmax[np.arange(len(x)), y] -= 1.
        gred = (1. / self.num_data) * np.dot(x.T, softmax)
        self.theta -= self.lr * gred
        print('theta:\n', self.theta)

if __name__ == '__main__':


        softmax_reg = softmax_regression()
        # loading the data
        py = osp.join("data", "exam_y.dat")
        px = osp.join("data", "exam_x.dat")
        iris_data = np.loadtxt(px)
        iris_label = np.loadtxt(py, dtype=int)
        data = iris_data
        label = iris_label
        data = softmax_reg.normalize(data)
        data = np.vstack((data.T, softmax_reg.bias)).T

        print('initiated weight is:\n', softmax_reg.weight)

        loss_list = []
        step_list = []
        acc_list = []
        plt.ion()
        fig, ax = plt.subplots(1, 4, figsize=(16, 5))
        for steps in range(300):
            step_list.append(steps)
            pred = softmax_reg.softmax(data)
            classification = np.argmax(pred, 1)
            loss = softmax_reg.loss(data,label)
            print('current loss is:\n', loss)
            loss_list.append(loss)

            plt.subplot(1, 4, 1)
            plt.title('ground truth')
            for i in range(self.num_class):
                data_x = np.array(data.T[0][label == i])
                data_y = np.array(data.T[1][label == i])
                plt.scatter(data_x, data_y, c=softmax_reg.color_list[i])

            plt.subplot(1, 4, 2)
            plt.title('classification scatter plot')
            for i in range(self.num_class):
                data_x = np.array(data.T[0][classification == i])
                data_y = np.array(data.T[1][classification == i])
                if len(data_x) == 0:
                    continue
                plt.scatter(data_x, data_y, c=softmax_reg.color_list[i])
            ax[1].cla()
            plt.subplot(1, 4, 3)
            plt.title('loss')
            ax[2].cla()
            plt.plot(step_list, loss_list, c='b', ls='-', marker='o')
            plt.subplot(1, 4, 4)
            acc = sum(label == classification) / softmax_reg.num_data
            acc_list.append(acc)
            plt.plot(step_list, acc_list, c='g', ls='-', marker='*')
            plt.title('accuracy')
            plt.pause(0.1)
            
            softmax_reg.update_parameter()
















