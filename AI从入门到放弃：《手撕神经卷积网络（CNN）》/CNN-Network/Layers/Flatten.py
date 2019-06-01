# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


class FlattenLayer(object):
    def __init__(self, upper_layer, name = ''):
        self.name = name
        self.type = 'flatten'
        self.upper_layer = upper_layer
        self.show_and_save_images = False

        self.batch_size = self.upper_layer.batch_size

        self.c_input, self.h_input, self.w_input = self.upper_layer.output_shape()

        self.w_output = 1
        self.h_output = 1
        self.c_output = self.c_input * self.h_input * self.w_input

        self.input_data = None
        self.output_data = None
        self.diff = None

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size
        self.output_data = self.input_data.reshape(self.input_data.shape[0], -1)
        self.show_forward_img(self.output_data)
        return self.output_data

    def backward(self, diff):
        self.diff = diff.reshape(self.input_data.shape)
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

    def output_data(self):
        return self.output_data

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)

    def load_params(self, weight, bias):
        pass

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable


class UpperLayerForFlattenTest(object):
    def __init__(self):
        self.batch_size = 1
        self.c_input = 1
        self.h_input = 4
        self.w_input = 4

        self.output_data = np.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]]])

        self.c_output, self.w_output, self.h_output = self.output_data.shape[1:]

    def forward(self):
        return self.output_data

    def backward(self, diff):
        pass

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)


if __name__ == '__main__':
    upper_layer = UpperLayerForFlattenTest()

    flatten_input = upper_layer.forward()
    print("\ninput to flatten layer:\n%s,\nshape: %s\n" % (flatten_input, flatten_input.shape))

    flatten = FlattenLayer(upper_layer)

    # 测试前向传播
    flatten_output = flatten.forward()
    expect_forward = np.array([[[
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]]])

    print("flatten forward expect output:\n%s,\nshape: %s\n" % (expect_forward, expect_forward.shape))
    print("flatten forward output:\n%s,\nshape: %s\n" % (flatten_output, flatten_output.shape))

    # 测试后向传播
    diff_next = np.array([[[
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]]])

    print("flattene diff:\n%s,\nshape: %s\n" % (diff_next, diff_next.shape))

    flatten.backward(diff_next)

    expect_backward = np.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
    ]]])
    print("flatten backward, expect diff:\n%s,\nshape: %s\n" % (expect_backward, expect_backward.shape))
    print("flatten backward, diff:\n%s,\nshape: %s\n" % (flatten.diff, flatten.diff.shape))

