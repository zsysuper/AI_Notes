#coding:utf-8

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

class MnistInput:
    def __init__(self, data_dir = "", batch_size = 32, test_size = 100, name = ''):
        self.name = name
        self.type = 'input'
        self.train_batch_count = 0
        self.test_batch_count = 0

        self.train_num = batch_size
        self.test_num = test_size

        self.batch_size = self.train_num

        self.c_output = 1  # 输出通道数，黑白图像颜色通道是1

        self.train_images = self.read_images(os.path.join(data_dir, 'train-images-idx3-ubyte')) / 256.
        self.train_labels = self.read_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))

        self.test_images = self.read_images(os.path.join(data_dir, 't10k-images-idx3-ubyte')) / 256.
        self.test_labels = self.read_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

        self.train_img_num, self.h_output, self.w_output = self.train_images.shape
        self.test_img_num, _, _ = self.test_images.shape

        self.train_cur_index = 0
        self.test_cur_index = 0

        self.output_images = None
        self.output_labels = None

        self.sets = 'train'

        self.batch_num_per_epoch = len(self.train_labels)/self.batch_size

        self.show_and_save_images = False

    def set_train_data(self):
        self.batch_size = self.train_num
        if self.train_cur_index + self.batch_size >= self.train_img_num:
            idx1 = np.arange(self.train_cur_index, self.train_img_num)
            idx2 = np.arange(0, self.train_cur_index + self.batch_size - self.train_img_num)
            output_index = np.append(idx1, idx2)

            self.train_batch_count = self.train_batch_count + 1
            self.train_cur_index = self.train_cur_index + self.batch_size - self.train_img_num
        else:
            output_index = np.arange(self.train_cur_index, self.train_cur_index + self.batch_size)
            self.train_cur_index = self.train_cur_index + self.batch_size

        self.output_images = self.train_images[output_index].reshape(self.batch_size, self.c_output, self.h_output,
                                                                     self.w_output)
        self.output_labels = self.train_labels[output_index].reshape(-1)
        return (self.output_images, self.output_labels)

    def set_test_data(self):
        self.batch_size = self.test_num
        if self.test_cur_index + self.batch_size >= self.test_img_num:
            idx1 = np.arange(self.test_cur_index, self.test_img_num)
            idx2 = np.arange(0, self.test_cur_index + self.batch_size - self.test_img_num)
            output_index = np.append(idx1, idx2)

            self.test_batch_count = self.test_batch_count = + 1
            self.test_cur_index = self.test_cur_index + self.batch_size - self.test_img_num
        else:
            output_index = np.arange(self.test_cur_index, self.test_cur_index + self.batch_size)
            self.test_cur_index = self.test_cur_index + self.batch_size

        self.output_images = self.test_images[output_index].reshape(self.batch_size, self.c_output, self.h_output,
                                                                    self.w_output)
        self.output_labels = self.test_labels[output_index].reshape(-1)

        return (self.output_images, self.output_labels)

    def forward(self):
        if self.sets == 'train':
            self.set_train_data()

        elif self.sets == 'test':
            self.set_test_data()
        else:
            return None

        self.show_forward_img(self.output_images)
        return self.output_images

    def backward(self, diff):
        pass

    def read_images(self, bin_img_f):
        f = open(bin_img_f, 'rb')
        buf = f.read()
        head = struct.unpack_from('>IIII', buf, 0)
        offset = struct.calcsize('>IIII')
        img_num = head[1]
        img_width = head[2]
        img_height = head[3]
        bits_size = img_num * img_height * img_width
        raw_imgs = struct.unpack_from('>' + str(bits_size) + 'B', buf, offset)
        f.close()
        imgs = np.reshape(raw_imgs, head[1:])
        return imgs

    def read_labels(self, bin_img_f):
        f = open(bin_img_f, 'rb')
        buf = f.read()
        head = struct.unpack_from('>II', buf, 0)
        img_num = head[1]
        offset = struct.calcsize('>II')
        raw_labels = struct.unpack_from('>' + str(img_num) + 'B', buf, offset)
        f.close()
        labels = np.reshape(raw_labels, [img_num, 1])
        return labels

    def show_imgs(self, show_num):
        imgs_data = self.train_images
        pic_wh = int(np.sqrt(show_num))
        plt.figure(figsize=(pic_wh, pic_wh))
        plt.subplots_adjust(wspace=0, hspace=0.15)
        for i in range(show_num):
            plt.subplot(show_num/10, 10, i + 1)
            plt.imshow(imgs_data[i], interpolation='none', cmap='binary')
            plt.xticks([])
            plt.yticks([])
        plt.savefig("OutputImages/train_pics.png")
        plt.show()

    def show_forward_img(self, imgs):
        if not self.show_and_save_images:
            return
        imgs_data = imgs
        show_num = imgs_data.shape[0]
        pic_wh = show_num
        plt.figure(figsize=(pic_wh, pic_wh))
        for i in range(show_num):
            plt.subplot(show_num, 1, i + 1)
            plt.imshow(imgs_data[i][0], interpolation='none', cmap='binary')
            plt.xticks([])
            plt.yticks([])

        #plt.suptitle("%s_output" % self.name, fontproperties = font, fontsize=8)
        plt.savefig("OutputImages/%s_output.png" % self.name)
        plt.show()

    def output_shape(self):
        return (self.c_output, self.w_output, self.h_output)

    def get_images(self):
        return self.output_images

    def get_labels(self):
        return self.output_labels

    def get_batch_setting(self):
        return self.batch_size, self.test_num

    def choose_dataset(self, set = 'train'):
        if set.lower() != "train" and set.lower() != "test":
            print("Data Set Must Be 'train' or 'test', Set It With 'train' Default.")
            self.sets = 'train'
            return
        self.sets = set.lower()

    def sample_info(self):
        return (len(self.train_labels), len(self.test_labels), self.batch_num_per_epoch)

    def load_params(self, weight, bias):
        pass

    def set_show_and_save_images(self, enable = False):
        self.show_and_save_images = enable