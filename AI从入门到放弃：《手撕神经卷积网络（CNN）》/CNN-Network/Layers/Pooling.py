# coding:utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


class PoolingLayer(object):
    def __init__(self, upper_layer, k = 2, s = 2, name = ''):
        self.name = name
        self.type = 'pool'
        self.upper_layer = upper_layer
        self.show_and_save_images = False

        self.batch_size = self.upper_layer.batch_size

        self.c_input, self.h_input, self.w_input = self.upper_layer.output_shape()

        self.k = k
        self.s = s

        self.w_output = int(math.ceil((self.w_input - self.k + 0.) / self.s)) + 1
        self.h_output = int(math.ceil((self.h_input - self.k + 0.) / self.s)) + 1
        self.c_output = self.c_input

        self.input_data = None
        self.output_data = None
        self.max_index = None
        self.diff = None

    def forward(self):
        '''
        举例：
        input_data = [[[
            [0   1   2   3   4   5 ]
            [6   7   8   9   10  11]
            [12  13  14  15  16  17]
            [18  19  20  21  22  23]
            [24  25  26  27  28  29]
            [30  31  32  33  34  35]
        ]]]

        池化核大小2，步长2
        input_col = im2col(input_data, 2, 2)
                  = [[ 0  1  6  7]
                     [ 2  3  8  9]
                     [ 4  5 10 11]
                     [12 13 18 19]
                     [14 15 20 21]
                     [16 17 22 23]
                     [24 25 30 31]
                     [26 27 32 33]
                     [28 29 34 35]]
        很明显的，当kernerl_size=2, stride=2, 输入为（6x6）时，切割出的池化区域的数量为9，每个池化区大小为4，所以
        input_col.shape = (9, 4)

        生成input_col的下标矩阵
        temp_index = np.tile(np.arange(input_col.shape[1]),input_col.shape[0]).reshape(input_col.shape)
                   = [
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                         [0 1 2 3]
                     ]

        标记出每个池化区的最大元素的位置
        max_index = tmp_index == input_col.argmax(1).reshape(-1,1)
                  = [
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                         [False False False  True]
                    ]

        池化区个数时9，那么输出就是(3x3)
        output_data = input_col[max_index].reshape(num, num_input, output_h, output_w)
                    = input_col[max_index].reshape(1, 1, 3, 3)
                    = [[[
                            [ 7  9 11]
                            [19 21 23]
                            [31 33 35]
                      ]]]

        max_index 被保存起来，池化成做反向传播的时候用
        '''
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size

        input_col = self.im2col(self.input_data, self.k, self.s)
        tmp_index = np.tile(np.arange(input_col.shape[1]),input_col.shape[0]).reshape(input_col.shape)
        self.max_index = tmp_index == input_col.argmax(1).reshape(-1,1)

        self.output_data = input_col[self.max_index].reshape(self.batch_size, self.c_output, self.h_output, self.w_output)
        self.show_forward_img(self.output_data)
        return self.output_data

    def backward(self, diff):
        '''
        :param diff: 上一层误差
        :return: 池化层误差

        举例：
        diff = [[[
                [ 7  9 11]
                [19 21 23]
                [31 33 35]
            ]]]

        diff_col = np.zeros((1 * 1 * 3 * 3, 2**2))
                 = [
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                         [0. 0. 0. 0.]
                    ]

        diff_col[self.max_index] = diff.reshape(-1)

        diff_col = [
                         [ 0.  0.  0.  7.]
                         [ 0.  0.  0.  9.]
                         [ 0.  0.  0. 11.]
                         [ 0.  0.  0. 19.]
                         [ 0.  0.  0. 21.]
                         [ 0.  0.  0. 23.]
                         [ 0.  0.  0. 31.]
                         [ 0.  0.  0. 33.]
                         [ 0.  0.  0. 35.]
                    ]

        diff = col2ims(diff_col, input_data.shape, kernel_size, stride)
             = col2ims(diff_col, (1,1,6,6), 2, 2)
             = [[[
                    [ 0.  0.  0.  0.  0.  0.]
                    [ 0.  7.  0.  9.  0. 11.]
                    [ 0.  0.  0.  0.  0.  0.]
                    [ 0. 19.  0. 21.  0. 23.]
                    [ 0.  0.  0.  0.  0.  0.]
                    [ 0. 31.  0. 33.  0. 35.]
               ]]]

        '''
        diff_col = np.zeros((self.batch_size * self.c_output * self.h_output * self.w_output, self.k**2))
        diff_col[self.max_index] = diff.reshape(-1)
        self.diff = self.col2ims(diff_col, self.input_data.shape, self.k, self.s)
        self.upper_layer.backward(self.diff)

    def show_forward_img(self, imgs):
        if not self.show_and_save_images:
            return
        imgs_data = imgs
        batch_size = imgs_data.shape[0]
        channel = imgs_data.shape[1]
        pic_wh = int(channel)
        plt.figure(figsize=(pic_wh, pic_wh))

        imgs_data = imgs_data.transpose(1, 0, 2, 3)
        imgs_data = imgs_data.reshape(batch_size * channel, imgs_data.shape[2], imgs_data.shape[3])

        for i in range(batch_size * channel):
            plt.subplot(channel, batch_size, i+1)
            plt.imshow(imgs_data[i], interpolation='none', cmap='binary')
            plt.xticks([])
            plt.yticks([])

        #plt.suptitle("%s_output" % self.name, fontproperties = font, fontsize=8)
        plt.savefig("OutputImages/%s_output.png" % self.name)
        plt.show()

    def output_data(self):
        return self.output_data

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)

    def im2col(self, X, kernel_size=1, stride=1):
        '''
        把图片按照卷积区切割后堆叠
        :param x: 图像输入
        :param kernel_size: 卷积核尺寸
        :param padding: 边界填充大小
        :param stride: 卷积步长
        :return: 图片按照卷积区切割后堆叠后的数据，用来和卷积核做卷积运算
        '''
        num, channels, height, width = X.shape
        surplus_height = (height - kernel_size) % stride
        surplus_width = (width - kernel_size) % stride
        pad_h = (kernel_size - surplus_height) % kernel_size
        pad_w = (kernel_size - surplus_width) % kernel_size
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        channel_idx, raw_idx, col_idx = self.im2col_indexes(X.shape, kernel_size, stride=stride)
        X_col = X[:, channel_idx, raw_idx, col_idx].reshape(num * channels, kernel_size ** 2, -1)
        X_col = X_col.transpose(0, 2, 1)
        return X_col.reshape(-1, kernel_size ** 2)

    def im2col_indexes(self, x_shape, kernel_size, padding=0, stride=1):
        '''
        :param x_shape: 输入图像的尺寸参数：通道数, 宽度, 高度
        :param kernel_size: 卷积核大小
        :param padding: 边界填充大小
        :param stride: 步长
        :return: 图像按卷积区切割后堆叠的数据
        '''
        N, C, H, W = x_shape
        assert (H + 2 * padding - kernel_size) % stride == 0
        assert (W + 2 * padding - kernel_size) % stride == 0

        out_height = int((H + 2 * padding - kernel_size) / stride + 1)
        out_width = int((W + 2 * padding - kernel_size) / stride + 1)

        kernel_raw_idx = np.repeat(np.arange(kernel_size), kernel_size)
        kernel_raw_idx = np.tile(kernel_raw_idx, C)

        convregion_raw_idx = stride * np.repeat(np.arange(out_height), out_width)

        kernel_col_idx = np.tile(np.arange(kernel_size), kernel_size * C)
        convregion_col_idx = stride * np.tile(np.arange(out_width), out_height)

        raw_idx = kernel_raw_idx.reshape(-1, 1) + convregion_raw_idx.reshape(1, -1)
        col_indx = kernel_col_idx.reshape(-1, 1) + convregion_col_idx.reshape(1, -1)

        channel_idx = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)

        return (channel_idx.astype(int), raw_idx.astype(int), col_indx.astype(int))

    def col2ims(self, x, img_shape, kernel_size, stride):
        '''
        :param img_shape: 还原出的图像的大小：深度（RBG通道）, 宽度, 高度
        :param kernel_size: 卷积核尺寸
        :param padding: 边界填充大小
        :param stride: 步长
        :return: 还原后的图像数据
        '''
        x_row_num, x_col_num = x.shape
        img_n, img_c, img_h, img_w = img_shape
        o_h = int(math.ceil((img_h - kernel_size + 0.) / stride)) + 1
        o_w = int(math.ceil((img_w - kernel_size + 0.) / stride)) + 1
        assert img_n * img_c * o_h * o_w == x_row_num
        assert kernel_size ** 2 == x_col_num
        surplus_h = (img_h - kernel_size) % stride
        surplus_w = (img_w - kernel_size) % stride
        pad_h = (kernel_size - surplus_h) % stride
        pad_w = (kernel_size - surplus_w) % stride
        output_padded = np.zeros((img_n, img_c, img_h + pad_h, img_w + pad_w))
        x_reshape = x.reshape(img_n, img_c, o_h, o_w, kernel_size, kernel_size)
        for n in range(img_n):
            for i in range(o_h):
                for j in range(o_w):
                    output_padded[n, :, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size] = \
                        output_padded[n, :, i * stride: i * stride + kernel_size,
                        j * stride: j * stride + kernel_size] + \
                        x_reshape[n, :, i, j, ...]
        return output_padded[:, :, 0: img_h + pad_h, 0: img_w + pad_w]

    def load_params(self, weight, bias):
        pass

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable

class UpperLayerForPoolingTest(object):
    def __init__(self):
        self.batch_size = 1
        self.c_input = 1
        self.h_input = 6
        self.w_input = 6

        self.output_data = np.array([[[
            [0,1,2,3,4,5],
            [6,7,8,9,10,11],
            [12,13,14,15,16,17],
            [18,19,20,21,22,23],
            [24,25,26,27,28,29],
            [30,31,32,33,34,35]
        ]]])

        self.c_output, self.w_output , self.h_output = self.output_data.shape[1:]

    def forward(self):
        return self.output_data

    def backward(self, diff):
        pass

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)


if __name__ == '__main__':
    upper_layer = UpperLayerForPoolingTest()

    pooling_input = upper_layer.forward()
    print("\ninput to pooling layer:\n%s,\nshape: %s\n" % (pooling_input, pooling_input.shape))

    pooling = PoolingLayer(upper_layer, 2, 2)

    # 测试前向传播
    pooling_output = pooling.forward()

    expect_forward = np.array([[[
        [ 7,  9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]])
    print("pooling forward expect output:\n%s,\nshape: %s\n" % (expect_forward, expect_forward.shape))
    print("pooling forward output:\n%s,\nshape: %s\n" % (pooling_output, pooling_output.shape))

    # 测试后向传播
    diff_next = np.array([[[
        [7, 9, 11],
        [19, 21, 23],
        [31, 33, 35]
    ]]])
    print("pooling diff:\n%s,\nshape: %s\n" % (diff_next, diff_next.shape))

    pooling.backward(diff_next)

    expect_backward = np.array([[[
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  7.,  0.,  9.,  0., 11.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., 19.,  0., 21.,  0., 23.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., 31.,  0., 33.,  0., 35.]
    ]]])
    print("pooling backward, expect diff:\n%s,\nshape: %s\n" % (expect_backward, expect_backward.shape))
    print("pooling backward, diff:\n%s,\nshape: %s\n" % (pooling.diff, pooling.diff.shape))