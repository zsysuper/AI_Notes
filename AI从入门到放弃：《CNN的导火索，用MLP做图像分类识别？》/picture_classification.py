#coding:utf-8
import h5py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import *

font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

def load_Cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    mycat_dataset = h5py.File('datasets/my_cat_misu.h5', "r")
    mycat_set_x_orig = np.array(mycat_dataset["mycat_set_x"][:])
    mycat_set_y_orig = np.array(mycat_dataset["mycat_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    mycat_set_y_orig = mycat_set_y_orig.reshape((1, mycat_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, mycat_set_x_orig, mycat_set_y_orig,classes


def predict_by_modle(x, y, nn):

    m = x.shape[1]
    p = np.zeros((1,m))

    output, caches = nn.forward_propagation(x)

    for i in range(0, output.shape[1]):
        if output[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # 预测出来的结果和期望的结果比对，看看准确率多少：
    # 比如100张预测图片里有50张猫的图片，只识别出40张，那么识别率就是80%
    print(u"识别率: "  + str(np.sum((p == y)/float(m))))
    return np.array(p[0], dtype=np.int), (p==y)[0], np.sum((p == y)/float(m))*100


def save_imgs_to_h5file(h5_fname, x_label, y_label, img_paths_list, img_label_list):
    data_imgs = np.random.rand(len(img_paths_list), 64, 64, 3).astype('int')
    label_imgs = np.random.rand(len(img_paths_list), 1).astype('int')

    for i in range(len(img_paths_list)):
        data_imgs[i] = np.array(plt.imread(img_paths_list[i]))
        label_imgs[i] = np.array(img_label_list[i])

    f = h5py.File(h5_fname, 'w')
    f.create_dataset(x_label, data=data_imgs)
    f.create_dataset(y_label, data=label_imgs)
    f.close()

    return data_imgs, label_imgs

if __name__ == "__main__":
    # 图片label为1代表这是一张喵星人的图片，0代表不是
    #save_imgs_to_h5file('datasets/my_cat_misu.h5', 'mycat_set_x', 'mycat_set_y', ['misu.jpg'],[1])

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, mycat_set_x_orig, mycat_set_y_orig, classes = load_Cat_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    mycat_x_flatten = mycat_set_x_orig.reshape(mycat_set_x_orig.shape[0], -1).T
    train_set_x = train_x_flatten / 255.
    test_set_x = test_x_flatten / 255.
    mycat_set_x = mycat_x_flatten / 255.

    print(u"训练图片数量: %d" % len(train_set_x_orig))
    print(u"测试图片数量: %d" % len(test_set_x_orig))

    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(wspace=0,hspace=0.15)
    for i in range(len(train_set_x_orig)):
        plt.subplot(21,10, i+1)
        plt.imshow(train_set_x_orig[i],interpolation='none',cmap='Reds_r',vmin=0.6,vmax=.9)
        plt.xticks([])
        plt.yticks([])
    plt.savefig("cat_pics_train.png")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    for i in range(len(test_set_x_orig)):
        ax = plt.subplot(8, 8, i + 1)
        im = ax.imshow(test_set_x_orig[i], interpolation='none', cmap='Reds_r', vmin=0.6, vmax=.9)
        plt.xticks([])
        plt.yticks([])

    plt.savefig("cat_pics_test.png")
    plt.show()

    plt.figure(figsize=(2, 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(mycat_set_x_orig)):
        ax = plt.subplot(1, 1, i + 1)
        im = ax.imshow(mycat_set_x_orig[i], interpolation='none', cmap='Reds_r', vmin=0.6, vmax=.9)
        plt.xticks([])
        plt.yticks([])

    plt.savefig("cat_pics_my.png")
    plt.show()

    # 用训练图片集训练模型
    layers_dims = [12288, 20, 7, 5, 1]
    nn = NeuralNetwork(layers_dims, True)
    nn.set_xy(train_set_x, train_set_y_orig)
    nn.set_num_iterations(10000)
    nn.set_learning_rate(0.0075)
    nn.training_modle()

    # 结果展示说明：
    # 【识别正确】：
    #   1.原图是猫，识别为猫  --> 原图显示
    #   2.原图不是猫，识别为不是猫 --> 降低显示亮度

    # 【识别错误】：
    #   1.原图是猫，但是识别为不是猫 --> 标红显示
    #   2.原图不是猫， 但是识别成猫 --> 标红显示

    # 训练用的图片走一遍模型，观察其识别率
    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(wspace=0, hspace=0.15)

    pred_train, true, accuracy = predict_by_modle(train_set_x, train_set_y_orig, nn)

    for i in range(len(train_set_x_orig)):
        ax = plt.subplot(21, 10, i + 1)

        x_data = train_set_x_orig[i]
        if pred_train[i] == 0 and train_set_y_orig[0][i] == 0:
            x_data = x_data/5

        if true[i] == False:
            x_data[:, :, 0] = x_data[:, :, 0] + (255 - x_data[:, :, 0])

        im = plt.imshow(x_data,interpolation='none',cmap='Reds_r',vmin=0.6,vmax=.9)

        plt.xticks([])
        plt.yticks([])

    plt.suptitle(u"Num Of Pictrues: %d\n Accuracy: %.2f%%" % (len(train_set_x_orig), accuracy), y=0.92, fontsize=20)
    plt.savefig("cat_pics_train_predict.png")
    plt.show()

    # 不属于训练图片集合的测试图片，走一遍模型，观察其识别率
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.1)

    pred_test, true, accuracy = predict_by_modle(test_set_x, test_set_y_orig, nn)

    for i in range(len(test_set_x_orig)):
        ax = plt.subplot(8, 8, i + 1)

        x_data = test_set_x_orig[i]
        if pred_test[i] == 0 and test_set_y_orig[0][i] == 0:
            x_data = x_data/5

        if true[i] == False:
            x_data[:, :, 0] = x_data[:, :, 0] + (255 - x_data[:, :, 0])

        im = ax.imshow(x_data, interpolation='none', cmap='Reds_r', vmin=0.6, vmax=.9)

        plt.xticks([])
        plt.yticks([])

        plt.suptitle(u"Num Of Pictrues: %d\n Accuracy: %.2f%%" % (len(mycat_set_x_orig), accuracy), fontsize=20)
    plt.savefig("cat_pics_test_predict.png")
    plt.show()

    # 用我家主子的照骗，走一遍模型，观察其识别率，因为只有一张图片，所以识别率要么 100% 要么 0%
    plt.figure(figsize=(2, 2.6))
    plt.subplots_adjust(wspace=0, hspace=0.1)

    pred_mycat, true, accuracy = predict_by_modle(mycat_set_x, mycat_set_y_orig, nn)
    for i in range(len(mycat_set_x_orig)):
        ax = plt.subplot(1, 1, i+1)

        x_data = mycat_set_x_orig[i]
        if pred_mycat[i] == 0 and mycat_set_y_orig[0][i] == 0:
            x_data = x_data/5

        if true[i] == False:
            x_data[:, :, 0] = x_data[:, :, 0] + (255 - x_data[:, :, 0])

        im = ax.imshow(x_data, interpolation='none', cmap='Reds_r', vmin=0.6, vmax=.9)

        plt.xticks([])
        plt.yticks([])

        if pred_mycat[i] == 1:
            plt.suptitle(u"我：'我主子是喵星人吗？'\nA I :'是滴'", fontproperties = font)
        else:
            plt.suptitle(u"我：'我主子是喵星人吗？'\nA I :'唔系~唔系~'", fontproperties = font)
    plt.savefig("cat_pics_my_predict.png")
    plt.show()