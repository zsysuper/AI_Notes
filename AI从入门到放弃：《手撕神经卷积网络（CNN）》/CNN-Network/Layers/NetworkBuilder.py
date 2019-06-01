# coding:utf-8
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from Layers.MnistInput import MnistInput
from Layers.Convolution import ConvLayer
from Layers.Pooling import PoolingLayer
from Layers.Relu import ReLuLayer
from Layers.Flatten import FlattenLayer
from Layers.FullConnection import FullConnectionLayer
from Layers.SoftmaxLoss import SoftmaxLossLayer, SoftmaxLayer

font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

class CNN_Network(object):
    def __init__(self, layer_list, base_lr = 1e-4, l2_regulation = 4e-6, epoch_period = 5):
        self.layers = []

        self.base_lr = np.float32(base_lr)
        self.lr = np.float32(base_lr)
        self.l2_regulation = np.float32(l2_regulation)

        self.epoch_period = epoch_period
        self.epoch_count = 1

        self.input_layer = None
        self.output_layer = None

        self.avg_loss_list = list()
        self.accuracy_list = list()

        self.batch_size = 0
        self.test_size = 0

        self.train_data_total_num = 0
        self.test_data_total_num = 0

        self.batch_size_per_epoch = 0

        self.stop_when_epoches_gte  = 5
        self.stop_when_accuracy_gte = 0.995

        self.show_and_save_images = False

        self.dataset = 'train'

        for i in range(0, len(layer_list)):
            layer = layer_list[i]

            # 第一层是输入层
            if i == 0 and layer['type'] != "input_layer":
                print("Type Of First Layer Must Be 'input_layer'")
                self.layers = []
                break
            elif i == 0 and layer['type'] == "input_layer":
                layer.pop('type')
                self.input_layer = MnistInput(**layer)
                self.layers.append(self.input_layer)
                self.batch_size, self.test_size = self.input_layer.get_batch_setting()
                self.test_data_total_num, self.test_data_total_num, self.batch_size_per_epoch = self.input_layer.sample_info()
                print(self.batch_size, self.test_size)
                print(self.train_data_total_num, self.test_data_total_num, self.batch_size_per_epoch)
                continue

            if i == len(layer_list) - 1 and layer['type'] != "softmax_loss":
                print("Type Of First Layer Must Be 'input_layer'")
                self.layers = []
                break
            elif i == len(layer_list) - 1 and layer['type'] == "softmax_loss":
                layer.pop('type')
                self.output_layer = SoftmaxLossLayer(self.layers[i-1], self.layers[0], **layer)
                self.layers.append(self.output_layer )
                continue

            if layer['type'] == 'conv':
                layer.pop('type')
                self.layers.append(ConvLayer(self.layers[i-1], **layer))

            elif layer['type'] == 'pool':
                layer.pop('type')
                self.layers.append(PoolingLayer(self.layers[i-1], **layer))

            elif layer['type'] == 'flatten':
                layer.pop('type')
                self.layers.append(FlattenLayer(self.layers[i-1], **layer))

            elif layer['type'] == 'relu':
                layer.pop('type')
                self.layers.append(ReLuLayer(self.layers[i-1], **layer))

            elif layer['type'] == 'fc':
                layer.pop('type')
                self.layers.append(FullConnectionLayer(self.layers[i-1], **layer))

            else:
                print("Unknow Layer-Type, Skip!")

    def show_data_images(self, num):
        self.input_layer.show_imgs(num)

    def show_network_info(self):
        layer_name_list = []
        for layer in self.layers:
            layer_name_list.append(layer.name)
        print("\n")
        print(' => '.join(layer_name_list))
        print("\n")

        print(u"训练集图片总数: %d" % self.train_data_total_num)
        print(u"测试集图片总数: %d" % self.test_data_total_num)
        print(u"batch_size: %d" % self.batch_size)
        print(u"一次迭代(epoch)需要训练的 batch 数: %d" % self.batch_size_per_epoch)

    def choose_dataset(self, dataset):
        if dataset != 'test' and dataset != 'train':
            print(u'数据集名称必须为：test/train')
            sys.exit(-1)
        self.dataset = dataset

    def predict(self):
        print(u"测试 ...")
        self.input_layer.choose_dataset(self.dataset)
        accuracy = self.calc_accuracy(self.output_layer.predict(), self.input_layer.get_labels())
        self.accuracy_list.append(accuracy)
        print(u"迭代次数: %d\t准确率 = %.2f%%" % (self.epoch_count, accuracy*100))
        return accuracy

    def train(self, epoch = 5):
        avg_loss = -1
        for i in range(epoch*self.batch_size_per_epoch):
            self.lr = self.base_lr / (1 + 1e-4 * i) ** 0.75

            #if i % self.batch_size_per_epoch == 0:
            # 训练集太大了，这里让他早点结束，相当于跳过一些训练数据，当然也可以使用上面的语句
            if i % 100 == 0:
                accuracy = self.predict()
                if accuracy >= self.stop_when_accuracy_gte and self.epoch_count >= self.stop_when_epoches_gte:
                    self.snapshot('Model/model.h5')
                    print(u"模型参数已经保存！")
                    plt.plot(np.arange(0, len(self.accuracy_list), 1), self.accuracy_list)
                    plt.xlabel(u'迭代次数', fontproperties=font)
                    plt.ylabel(u'准确率', fontproperties=font)
                    plt.title(u'CNN训练准确率曲线', fontproperties=font)
                    plt.grid(True)
                    plt.show()
                    plt.plot(np.arange(0, len(self.avg_loss_list), 1), self.avg_loss_list)
                    plt.xlabel(u'迭代次数', fontproperties=font)
                    plt.ylabel(u'误差值', fontproperties=font)
                    plt.title(u'CNN训练误差值曲线', fontproperties=font)
                    plt.grid(True)
                    plt.show()
                    break
            self.epoch_count += 1

            self.input_layer.choose_dataset(self.dataset)
            self.output_layer.forward()
            if avg_loss == -1:
                avg_loss = self.output_layer.get_loss()
            else:
                avg_loss = avg_loss * 0.9 + 0.1 * self.output_layer.get_loss()

            self.avg_loss_list.append(avg_loss)

            print(u"batch计数 = %-5d\t平均误差 = %.4f\t 学习率 = %f" % (i + 1, avg_loss, self.lr))
            self.output_layer.backward(self.lr)

    def run(self, stop_when_epoches_gte = 5, stop_when_accuracy_gte = 0.995):
        self.stop_when_epoches_gte = stop_when_epoches_gte
        self.stop_when_accuracy_gte = stop_when_accuracy_gte
        self.train(self.epoch_period)

    def calc_accuracy(self, prediction, truth):
        n = np.size(truth)
        return np.sum(prediction.argmax(1) == truth.reshape(-1)) / (n + 0.)

    def snapshot(self, model_fname):
        f = h5py.File(model_fname, "w")
        for layer in self.layers:
            if layer.type == 'conv' or layer.type == 'fc':
                g = f.create_group(layer.name)
                g.create_dataset('w', data = layer.weight)
                g.create_dataset('b', data = layer.bias)
        f.close()

    def load_model(self, model_fname):
        data_set = h5py.File(model_fname, "r")
        for layer in self.layers:
            if layer.name in data_set.keys():
                w = data_set[layer.name]['w'][:]
                b = data_set[layer.name]['b'][:]
                layer.load_params(w, b)

    def set_show_and_save_images(self, enable = False):
        for layer in self.layers:
            layer.set_show_and_save_images(enable)