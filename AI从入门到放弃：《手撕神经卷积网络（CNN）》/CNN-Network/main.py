# coding:utf-8


from Layers.NetworkBuilder import CNN_Network


if __name__ == '__main__':
    print('CNN Network Building...')
    cnn = CNN_Network([
        {'name': 'mnist_data',  'type': 'input_layer', 'data_dir': 'data/mnist', 'batch_size': 64, 'test_size': 1},
        {'name': 'conv1',       'type': 'conv', 'kn': 20, 'k': 5, 's': 1, 'p': 0},
        {'name': 'pool1',       'type': 'pool', 'k': 2, 's': 2},
        {'name': 'conv2',       'type': 'conv', 'kn': 20, 'k': 5, 's': 1, 'p': 0},
        {'name': 'pool2',       'type': 'pool', 'k': 2, 's': 2},
        {'name': 'flatten',     'type': 'flatten'},
        {'name': 'fc1',         'type': 'fc', 'c_output': 500},
        {'name': 'relu1',       'type': 'relu'},
        {'name': 'fc2',         'type': 'fc', 'c_output': 10},
        {'name': 'softmax_loss','type': 'softmax_loss'}

    ], base_lr = 1e-2, l2_regulation = 8e-6, epoch_period = 10)

    # 显示cnn网络结构信息
    cnn.show_network_info()

    # 挑前n张训练集合的图片显示
    #cnn.show_data_images(100)

    # 训练打开下面3行代码
    #cnn.choose_dataset('train')
    #cnn.set_show_and_save_images(enable=False)
    #cnn.run(stop_when_epoches_gte = 9000, stop_when_accuracy_gte = 0.995)

    # 用现成模型测试打开下面4行代码
    cnn.choose_dataset('test')
    cnn.load_model('Model/model.h5')
    cnn.set_show_and_save_images(enable=True)
    cnn.predict()
