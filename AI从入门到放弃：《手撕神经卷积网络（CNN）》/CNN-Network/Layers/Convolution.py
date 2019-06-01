# coding:utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')


class ConvLayer(object):
    def __init__(self, upper_layer, kn = 1, k = 2, p = 0, s = 1, TEST = "", name = ''):
        '''
        :param input_shape: 图片尺寸参数：深度（RBG通道）, 宽度, 高度
        :param kn: 卷积核个数
        :param k: 卷积核尺寸
        :param s: 步长
        :param p: 边界填充大小
        '''
        self.name = name
        self.type = 'conv'
        self.upper_layer = upper_layer
        self.show_and_save_images = False

        self.batch_size = self.upper_layer.batch_size

        self.c_input, self.h_input,self.w_input = self.upper_layer.output_shape()

        self.kn = kn
        self.k = k
        self.s = s
        self.p = p

        self.w_output = int((self.w_input - self.k + 2 * self.p) / self.s + 1)
        self.h_output = int((self.h_input - self.k + 2 * self.p) / self.s + 1)
        self.c_output = self.kn

        if TEST == "TEST":
            self.weight = np.ones((self.kn, self.c_input, self.k, self.k))
            self.bias = np.zeros((self.kn))
        else:
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.upper_layer.output_shape()) / self.c_output)
            self.weight = np.random.standard_normal((self.kn, self.c_input, self.k, self.k)) / weights_scale
            self.bias = np.random.standard_normal(self.kn) / weights_scale

        self.weight_diff_history = np.zeros(self.weight.shape)
        self.bias_diff_history = np.zeros(self.bias.shape)
        self.diff = None

        self.input_data = None
        self.output_data = None

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.batch_size = self.upper_layer.batch_size

        X_col = self.im2col(self.input_data, self.k, padding=self.p, stride=self.s)
        W_col = self.weight.reshape(self.kn, -1)

        # 计算卷积: conv = X * W + b
        self.output_data = (np.dot(W_col, X_col).T + self.bias).T
        self.output_data = self.output_data.reshape(self.c_output, self.h_output, self.w_output, self.batch_size)
        self.output_data = self.output_data.transpose(3, 0, 1, 2)
        self.show_forward_img(self.output_data)
        return self.output_data

    def backward(self, diff):
        self.diff = np.zeros(self.input_data.shape)

        weight_diff = np.zeros(self.weight.shape)
        weight_diff = weight_diff.reshape(weight_diff.shape[0], -1)
        bias_diff = np.zeros((self.c_output))

        weight_raws = self.weight.reshape(self.weight.shape[0], -1).T

        for i in range(self.batch_size):
            # dW_(l) = a_(l-1) * diff_(l)
            input_data_col = self.im2col(self.input_data[[i]], self.k, self.p, self.s)
            weight_diff = weight_diff + diff[i].reshape(diff[i].shape[0], -1).dot(input_data_col.T)

            # db_(l) = sum(diff_(l))
            bias_diff = bias_diff + np.sum(diff[i].reshape(diff[i].shape[0], -1), 1)

            # diff_(l-1) = diff_(l) * rot180(W_(l))
            diff_cols = weight_raws.dot(diff[i].reshape(diff[i].shape[0], -1))
            self.diff[i, ...] = self.col2im(diff_cols, self.input_data.shape[1:], self.k, self.p, self.s)

        # 更新 W, b
        self.weight_diff_history = 0.9 * self.weight_diff_history + weight_diff.reshape(self.weight.shape)
        #self.weight_diff_history = weight_diff.reshape(self.weight.shape)
        self.weight = self.weight * 0.9995 - self.weight_diff_history

        self.bias_diff_history = 0.9 * self.bias_diff_history + 2 * bias_diff
        #self.bias_diff_history = bias_diff
        self.bias = self.bias * 0.9995 - self.bias_diff_history

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

    def im2col(self, x, kernel_size, padding=0, stride=1):
        '''
        把图片按照卷积区切割后堆叠
        :param x: 图像输入
        :param kernel_size: 卷积核尺寸
        :param padding: 边界填充大小
        :param stride: 卷积步长
        :return: 图片按照卷积区切割后堆叠后的数据，用来和卷积核做卷积运算
        '''
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        channel_idx, raw_idx, col_idx = self.im2col_indexes(x.shape, kernel_size, padding, stride)
        cols = x_padded[:, channel_idx, raw_idx, col_idx]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(kernel_size ** 2 * C, -1)
        return cols

    def col2im(self, x, img_shape, kernel_size, padding=0, stride=1):
        '''
        :param img_shape: 还原出的图像的大小：深度（RBG通道）, 宽度, 高度
        :param kernel_size: 卷积核尺寸
        :param padding: 边界填充大小
        :param stride: 步长
        :return: 还原后的图像数据

         该函数用在卷积层反向传播时，把误差矩阵按照卷积运算的方式堆叠，就能得到卷积层的上一层误差
         举例:
         假设卷积层输入的图片大小是3x3，通道数为1，卷积核大小是2x2，那么卷积层输出的图片大小 2x2
         卷积层上一层传来的误差是4x4的（因为卷积核从左到右从上到下需要滑动4次，卷积核参数是2x2=4）
         x=[[1  2   3   4 ]
            [5  6   7   8 ]
            [9  10  11  12]
            [13 14  15  16]]

         x_row_num, x_col_num = x.shape =（4,4）
         channels, img_height, img_width = img_shape = （1, 3, 3）

         x_width = img_width - kernel_size + padding + 1 = 3 - 2 + 0 + 1 = 2
         x_height = img_height - kernel_size + padding + 1 =  3 - 2 + 0 + 1 = 2

         即卷积层输出大小是2x2的，卷积核是2x2的，所以按照卷积区域变换成5维矩阵，通道数是1
         x_reshape = x.T.reshape(x_height, x_width, channels, kernel_size, kernel_size)
                   = x.T.reshape(2, 2, 1, 2, 2)
                   = [
                        [
                            [
                              [
                                [ 1  5]
                                [ 9 13]
                              ]
                            ]

                            [
                              [
                                [ 2  6]
                                [10 14]
                              ]
                            ]
                        ]

                        [
                            [
                              [
                                [ 3  7]
                                [11 15]
                              ]
                            ]

                            [
                              [
                                [ 4  8]
                                [12 16]
                              ]
                            ]
                        ]
                   ]

         按照卷积运算方式，把卷积区域堆叠后，就还原出卷积层输入图像的大小，这种计算方式本身就满足了卷积核旋转180度的要求
         for i in range(x_height):
            for j in range(x_width):
                从左到右从上到下
                [[[0 0 0]
                  [0 0 0]
                  [0 0 0]]]

                [[[ 1  5  0]
                  [ 9 13  0]
                  [ 0  0  0]]]

                [[[ 1    5+2=7   6]
                  [ 9  13+10=23  14]
                  [ 0        0   0]]]

                [[[    1        7    6]
                  [9+3=12  23+7=30   14]
                  [    11       15   0]]]

                [[[ 1        7        6]
                  [12  30+4 =34  14+8=22]
                  [11  15+12=27       16]]]


         最后得到的 output_padded.shape = (1, 3, 3)
         output_padded = [[
                            [ 1  7   6 ]
                            [12  34  22]
                            [11  27  16]
                         ]]
         这样就求得了卷积层传播给上一层的误差矩阵 d_(l-1) = d_(l) * rot180[w_(l)]
        '''
        x_row_num, x_col_num = x.shape

        channels, img_height, img_width = img_shape

        x_width = int((img_width - kernel_size + 2*padding)/stride + 1)
        x_height = int((img_height - kernel_size + 2*padding)/stride + 1)

        assert channels * kernel_size ** 2 == x_row_num
        assert x_width * x_height == x_col_num

        x_reshape = x.T.reshape(x_height, x_width, channels, kernel_size, kernel_size)
        output_padded = np.zeros((channels, img_height + 2 * padding, img_width + 2 * padding))
        for i in range(x_height):
            for j in range(x_width):
                output_padded[:, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size] = \
                    output_padded[:, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size] + \
                    x_reshape[i, j, ...]
        return output_padded[:, padding: img_height + padding, padding: img_width + padding]

    def im2col_indexes(self, x_shape, kernel_size, padding=0, stride=1):
        '''
        :param x_shape: 输入图像的尺寸参数：通道数, 宽度, 高度
        :param kernel_size: 卷积核大小
        :param padding: 边界填充大小
        :param stride: 步长
        :return: 图像按卷积区切割后堆叠的数据

         解释一下这个函数的原理，这是卷积计算的核心
                1. 假设输入的数据x.shape=（1,1,6,6）
                           W (列)
                        o -------->
                        |         |
                      H |         | (行)
                        |         |
                         ---------
                    　
                x= [[[
                        [ 0  1  2  3  4  5]
                        [ 6  7  8  9 10 11]
                        [12 13 14 15 16 17]
                        [18 19 20 21 22 23]
                        [24 25 26 27 28 29]
                        [30 31 32 33 34 35]
                   ]]]

                   那么：  N, C, H, W = 1, 1, 6, 6
                          out_height = ((H + 2p -k)/s + 1) = ((6 + 2*0 - 2)/1 + 1) = 5
                          out_width = ((W + 2p -k)/s + 1) = ((6 + 2*0 - 2)/1 + 1) = 5

                2. 无padding的情况下 x_padded = x

                3. kernel大小是 2x2，那么卷积结果输出大小是 5x5

                4. 行(raw)位移索引矩阵
                    4.1 因为卷积核大小是2x2，所以行(raw)索引位移只跨两行，偏移量就是0或1
                        kernel_raw_idx = np.repeat(np.arange(kernel_size), kernel_size) = [0 0 1 1]

                    4.2 因为步长 stride = 1，而行数 out_height = 5。 所以卷积区域的行(raw)索引范围是 [0, 4],
                        复制和列数 out_width 相等的份数，因为卷积核向右滑动时需要滑动 right_slip_times = out_width 次
                        convregion_raw_idx = stride * np.repeat(np.arange(out_height), out_width)
                                           = [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]

                    4.3 得出行(raw)索引
                        偏移+卷积区域范围，因为卷积核在行(raw)的行为是从左到右滑动的
                        raw_idx = kernel_raw_idx.reshape(-1, 1) + convregion_raw_idx.reshape(1, -1)
                          = [
                                [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
                                [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
                                [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]
                                [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]
                            ]

                5. 列(column)位移索引矩阵
                    5.1 因为卷积核大小是2x2，通道数C=1，所以行索引位移只跨两两列，偏移量就是0或1
                        kernel_col_idx = np.tile(np.arange(kernel_size), kernel_size * C) = [0 1 0 1]

                    5.2 因为步长 stride = 1，而行数 out_width = 5。 所以卷积区域的列(column)索引范围是 [0, 4]，
                        并复制和行数 out_height 相等的份数，因为卷积核向下滑动时需要滑动 left_slip_times = out_height 次
                        convregion_col_idx = stride * np.tile(np.arange(out_width), out_height)
                                           = [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]

                    5.3  得出列(column)索引
                        col_indx = kernel_col_idx.reshape(-1, 1) + convregion_col_idx.reshape(1, -1)
                                 =[
                                        [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
                                        [1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5]
                                        [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
                                        [1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5]
                                  ]

                6. 利用行列索引矩阵，取得做卷积运算的所有卷积面数据，每个数据大小是  kernel_size*kernel_size = 2*2 = 4
                    怎么利用k,i,j呢？
                    因为通道C=1个，所以k始终是索引0下标的数据：
                    channel_idx = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)
                                = [
                                      [0]
                                      [0]
                                      [0]
                                      [0]
                                  ]

                    看输入数据：
                    x= [[[
                            [ 0  1  2  3  4  5]
                            [ 6  7  8  9 10 11]
                            [12 13 14 15 16 17]
                            [18 19 20 21 22 23]
                            [24 25 26 27 28 29]
                            [30 31 32 33 34 35]
                       ]]]

                    那么2x2的卷积核，第一次卷积的数据应该是 ：
                    [ 0  1 ]
                    [ 6  7 ]

                    这个数据怎么拿呢？记住padding=0，x_padded = x
                        行索引 raw_idx[0][0] = 0
                        列索引 col_indx[0][0] = 0
                        所以 x 的第一行第一列数据是 x_padded[:, 0, 0, 0] = 0

                        行索引 raw_idx[1][0] = 0
                        列索引 col_indx[1][0] = 1
                        所以 x 的第一行第二列数据是 x_padded[:, 0, 0, 1] = 1

                        行索引 raw_idx[2][0] = 1
                        列索引 col_indx[2][0] = 0
                        所以 x 的第二行第一列的数据是 x_padded[:, 0, 1, 0] = 6

                        行索引 raw_idx[3][0] = 1
                        列索引 col_indx[3][0] = 1
                        所以 x 的第二行第二列的数据是 x_padded[:, 0, 1, 1] = 7

                        所以cols的第一列数据是 0,1,6,7
                        同理计算 i[n][1],j[n][1],即移动raw_idx，col_indx的列，就相当于把卷积核从左到右从上到下把图像扫描了一遍


                    cols = x_padded[:, channel_idx, raw_idx, col_indx] = [[[
                        [ 0  1  2  3  4  6  7  8  9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28]
                        [ 1  2  3  4  5  7  8  9 10 11 13 14 15 16 17 19 20 21 22 23 25 26 27 28 29]
                        [ 6  7  8  9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28 30 31 32 33 34]
                        [ 7  8  9 10 11 13 14 15 16 17 19 20 21 22 23 25 26 27 28 29 31 32 33 34 35]
                    ]]]

                7. 为什么用这么复杂的方式呢，因为高效，用矩阵点乘就完成了卷积计算。简单粗暴的实现方式就是for循环，
                   按照步长左到右从上到下循环一遍。契合公式比较直观，但缺点就是慢
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

    def load_params(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable

class UpperLayerForConvTest(object):
    def __init__(self):
        self.batch_size = 2
        self.c_input = 3
        self.h_input = 3
        self.w_input = 3

        self.output_data = np.array([[[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ],[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ],[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ]],[[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ],[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ],[
            [1.,1.,1.],
            [1.,1.,1.],
            [1.,1.,1.]
        ]]])

        self.c_output, self.w_output , self.h_output = self.output_data.shape[1:]

    def forward(self):
        return self.output_data

    def backward(self, diff):
        pass

    def output_shape(self):
        return (self.c_output, self.h_output, self.w_output)

    def set_show_and_save_images(self, enable = False):
        print self.name, enable
        self.show_and_save_images = enable


if __name__ == '__main__':
    upper_layer = UpperLayerForConvTest()

    conv_input = upper_layer.forward()
    print("\ninput to conv layer:\n%s,\nshape: %s\n" % (conv_input, conv_input.shape))

    conv = ConvLayer(upper_layer, 1, 3, 1, 1, "TEST")

    # 测试前向传播
    conv_output = conv.forward()

    expect_forward = np.array([[[
        [4., 4.],
        [4., 4.]
    ]]])
    print("conv forward expect output:\n%s,\nshape: %s\n" % (expect_forward, expect_forward.shape))
    print("conv forward output:\n%s,\nshape: %s\n" % (conv_output, conv_output.shape))

    # 测试后向传播
    diff_next = np.array([[[
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
    ]], [[
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
    ]]])
    print("conv diff:\n%s,\nshape: %s\n" % (diff_next, diff_next.shape))

    conv.backward(diff_next)

    expect_backward = np.array([[[
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]
    ]],[[
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]
    ]],[[
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]
    ]]])
    print("conv backward, expect diff:\n%s,\nshape: %s\n" % (expect_backward, expect_backward.shape))
    print("conv backward, diff:\n%s,\nshape: %s\n" % (conv.diff, conv.diff.shape))
