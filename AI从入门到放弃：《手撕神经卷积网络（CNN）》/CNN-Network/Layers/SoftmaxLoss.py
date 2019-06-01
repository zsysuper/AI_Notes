# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


class SoftmaxLayer(object):
    def __init__(self, upper_layer, name = ''):
        self.name = name
        self.upper_layer = upper_layer
        self.show_and_save_images = False
        self.batch_size = self.upper_layer.batch_size

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size
        t = np.exp(self.input_data - self.input_data.max(1).reshape(-1, 1))
        self.output_data = t / t.sum(1).reshape(-1, 1)

        self.show_forward_img(self.output_data)
        return self.output_data

    def show_forward_img(self, imgs):
        if not self.show_and_save_images:
            return
        imgs_data = imgs
        channel = imgs_data.shape[0]
        pic_wh = int(channel)
        plt.figure(figsize=(pic_wh, imgs_data.shape[1]/10))
        for i in range(channel):
            plt.subplot(1, channel, i + 1)
            img = imgs_data[i].reshape(imgs_data.shape[1], 1)
            plt.imshow(img, interpolation='none', cmap='binary')
            plt.xticks([])
            plt.yticks([])
            #plt.suptitle("%s_output" % self.name, fontproperties = font, fontsize=8)
        plt.savefig("OutputImages/%s_output.png" % self.name)
        plt.show()

    def get_output(self):
        return self.output_data

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable


class SoftmaxLossLayer(object):
    def __init__(self, upper_layer, data_input_layer, name = ''):
        self.name = name
        self.type = 'softmax'
        self.upper_layer = upper_layer
        self.data_input_layer = data_input_layer

        self.batch_size = self.upper_layer.batch_size

        self.w_output = 1
        self.h_output = 1
        self.c_output = 1

        self.input_data = None
        self.output_data = None
        self.diff = None

        self.loss = 0

        self.softmax = SoftmaxLayer(upper_layer, name = self.name)

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size

        _, dim = self.input_data.shape

        # 减去最大值防止溢出
        t = np.exp(self.input_data - self.input_data.max(1).reshape(-1 ,1))
        softmax_data = t / t.sum(1).reshape(-1 ,1)

        # 太小的数置0
        softmax_data[softmax_data < 1e-30] = 1e-30

        s = np.tile(np.arange(dim), self.batch_size).reshape(self.input_data.shape)

        gt_index = s == self.data_input_layer.get_labels().reshape(-1, 1)

        # 根据标定值lable索引到输出位置，然后计算全局误差
        self.loss = 0 - np.average(np.log(softmax_data[gt_index]))

        self.diff = softmax_data.copy()

        self.output_data = softmax_data.copy()

        # d_softmax_loss = (softmax_output - 1) / m
        self.diff[gt_index] = self.diff[gt_index] - 1.
        self.diff = self.diff / self.batch_size
        return self.loss

    def backward(self, learning_rate):
        # 这里直接把学习率乘以误差梯度进行反向传播
        self.upper_layer.backward(self.diff * learning_rate)

    def show_forward_img(self, imgs):
        if not self.show_and_save_images:
            return
        imgs_data = imgs
        channel = imgs_data.shape[0]
        pic_wh = int(channel)
        plt.figure(figsize=(pic_wh, imgs_data.shape[1]/10))
        for i in range(channel):
            plt.subplot(1, channel, i + 1)
            img = imgs_data[i].reshape(imgs_data.shape[1], 1)
            plt.imshow(img, interpolation='none', cmap='binary')
            plt.xticks([])
            plt.yticks([])
            #plt.suptitle("%s_output" % self.name, fontproperties = font, fontsize=8)
        plt.savefig("OutputImages/%s_output.png" % self.name)
        plt.show()

    def predict(self):
        return self.softmax.forward()

    def get_output(self):
        return self.output_data

    def get_loss(self):
        return self.loss

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)

    def load_params(self, weight, bias):
        pass

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable
        self.softmax.set_show_and_save_images(enable)