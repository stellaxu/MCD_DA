import random
import torch
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable
import seperate_data
from basenet import *

def predict_loss(cls, y_pre): #requires how the loss is calculated for the preduct value and the ground truth value
    """
    Calculate the cross entropy loss for prediction of one picture
    :param y:
    :param y_pre:
    :return:
    """
    # cls_torch = np.full(1, cls)
    pre_cls_torch = y_pre.double()
    # target = torch.LongTensor([cls]).cuda()
    target = cls
    entropy = nn.CrossEntropyLoss()
    return entropy(pre_cls_torch, target)


def cross_validation_loss(args, feature_network_path, predict_network_path, num_layer, src_cls_list, target_path, val_cls_list, class_num, resize_size, crop_size, batch_size, use_gpu):
    """
    Main function for computing the CV loss
    :param feature_network:
    :param predict_network:
    :param src_cls_list:
    :param target_path:
    :param val_cls_list:
    :param class_num:
    :param resize_size:
    :param crop_size:
    :param batch_size:
    :return:
    """
    option = 'resnet' + args.resnet
    G = ResBase(option)
    F1 = ResClassifier(num_layer=num_layer)

    # F1.apply(weights_init)

    G.load_state_dict(torch.load(feature_network_path))
    F1.load_state_dict(torch.load(predict_network_path))

    if use_gpu:
        G.cuda()
        F1.cuda()
    G.eval()
    F1.eval()


    val_cls_list = seperate_data.dimension_rd(val_cls_list)
    prep_dict_val = prep.image_train(resize_size=resize_size, crop_size=crop_size)
    # load different class's image
    dsets_val = ImageList(val_cls_list, transform=prep_dict_val)
    dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)

    # prepare validation feature and predicted label for validation
    iter_val = iter(dset_loaders_val)
    val_input, val_labels = iter_val.next()
    if use_gpu:
        val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
    else:
        val_input, val_labels = Variable(val_input), Variable(val_labels)

    feature_val = G(val_input)
    pred_label = F1(feature_val)
    # _, pred_label = predict_network(val_input)

    # print(pred_label.shape)
    # print(pred_label)
    # print(pred_label[0])
    # print(val_labels[0])
    # print(val_labels[0] + 1)

    w = pred_label[0].shape[0]

    error = np.zeros(1)
    # print(predict_loss(val_labels[0], pred_label[0].view(1, w)))
    # print(predict_loss(val_labels[0], pred_label[0].view(1, w))[0])
    error[0] = predict_loss(val_labels[0], pred_label[0].view(1, w))[0]
    print("Error: {}".format(error[0]))
    error = error.reshape(1, 1)
    print(error)
    for num_image in range(1, len(pred_label)):
        new_error = np.zeros(1)

        single_pred_label = pred_label[num_image]
        w = single_pred_label.shape[0]
        single_val_label = val_labels[num_image]
        # print(single_val_label)
        # print(single_pred_label)
        # print(predict_loss(single_val_label, single_pred_label.view(1, w)))
        new_error[0] = predict_loss(single_val_label, single_pred_label.view(1, w))[0]
        new_error = new_error.reshape(1, 1)
        # print("New error: {}".format(new_error[0]))
        error = np.append(error, new_error, axis=0)
        # print("Error: {}".format(error))
        # print("Error size: {}".format(error.shape))

    for _ in range(len(iter_val) - 1):
        val_input, val_labels = iter_val.next()
        if use_gpu:
            val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
        else:
            val_input, val_labels = Variable(val_input), Variable(val_labels)
        feature_val = G(val_input)
        pred_label = F1(feature_val)
        # _, pred_label = predict_network(val_input)
        for num_image in range(len(pred_label)):
            new_error = np.zeros(1)

            single_pred_label = pred_label[num_image]
            w = single_pred_label.shape[0]
            single_val_label = val_labels[num_image]

            new_error[0] = predict_loss(single_val_label, single_pred_label.view(1, w))[0]
            new_error = new_error.reshape(1, 1)

            error = np.append(error, new_error, axis=0)
            # print("Error: {}".format(error))
            # print("Error size: {}".format(error.shape))
    print("Error: {}".format(error))
    cross_val_loss = error.sum()
    return cross_val_loss
