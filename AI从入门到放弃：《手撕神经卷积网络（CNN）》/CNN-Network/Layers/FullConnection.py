# coding:utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


class FullConnectionLayer(object):
    def __init__(self, upper_layer, c_output = 1, TEST = "", name = ''):
        self.name = name
        self.type = 'fc'
        self.upper_layer = upper_layer
        self.show_and_save_images = False

        self.batch_size = self.upper_layer.batch_size

        self.c_input, self.h_input, self.w_input = self.upper_layer.output_shape()

        self.w_output = self.w_input
        self.h_output = self.h_input
        self.c_output = c_output

        if TEST == "TEST":
            self.weight = np.ones((self.c_input, self.c_output))
            self.bias = np.zeros(self.c_output)
        else:
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, (self.c_input, self.c_output)) / self.c_output)
            self.weight = np.random.standard_normal((self.c_input, self.c_output)) / weights_scale
            self.bias = np.random.standard_normal(self.c_output) / weights_scale

        self.weight_diff_history = np.zeros(self.weight.shape)
        self.bias_diff_history = np.zeros(self.bias.shape)
        self.diff = None

        self.input_data = None
        self.output_data = None

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size

        input_cols = self.input_data.reshape(self.input_data.shape[0], -1)
        self.output_data = input_cols.dot(self.weight) + self.bias

        self.show_forward_img(self.output_data)
        return self.output_data

    def backward(self, diff):
        weight_diff = self.input_data.T.dot(diff)
        bias_diff = np.sum(diff, axis = 0)

        self.diff = diff.dot(self.weight.T)

        self.weight_diff_history = 0.9 * self.weight_diff_history + weight_diff
        #self.weight_diff_history = weight_diff
        self.weight = self.weight * 0.9995 - self.weight_diff_history

        self.bias_diff_history = 0.9 * self.bias_diff_history + 2 * bias_diff
        #self.bias_diff_history = bias_diff
        self.bias = self.bias * 0.9995 - self.bias_diff_history

        self.upper_layer.backward(self.diff)

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

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)

    def load_params(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable


class UpperLayerForFullconnectionTest(object):
    def __init__(self):
        self.batch_size = 1
        self.c_input = 4
        self.h_input = 1
        self.w_input = 1

        self.output_data = np.array([
            [1, 2, 3, 4]
        ])

        self.c_output = self.c_input
        self.w_output = 1
        self.h_output = 1

    def forward(self):
        return self.output_data

    def backward(self, diff):
        pass

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)


if __name__ == '__main__':
    upper_layer = UpperLayerForFullconnectionTest()

    full_connection_input = upper_layer.forward()
    print("\ninput to full_connection layer:\n%s,\nshape: %s\n" % (full_connection_input, full_connection_input.shape))

    full_connection = FullConnectionLayer(upper_layer, c_output = 2, TEST = "TEST")

    # 测试前向传播
    full_connection_output = full_connection.forward()
    expect_forward = np.array([
            [10, 10]
    ])

    print("full_connection forward expect output:\n%s,\nshape: %s\n" % (expect_forward, expect_forward.shape))
    print("full_connection forward output:\n%s,\nshape: %s\n" % (full_connection_output, full_connection_output.shape))

    # 测试后向传播
    diff_next = np.array([
            [1, 2]
    ])
    print("full_connectione diff:\n%s,\nshape: %s\n" % (diff_next, diff_next.shape))

    full_connection.backward(diff_next)

    expect_backward = np.array([
            [3, 3, 3, 3]
    ])

    print("full_connection backward, expect diff:\n%s,\nshape: %s\n" % (expect_backward, expect_backward.shape))
    print("full_connection backward, diff:\n%s,\nshape: %s\n" % (full_connection.diff, full_connection.diff.shape))

