import random
import torch
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch.utils.data as util_data
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable
import seperate_data


def predict_loss(cls, y_pre): #requires how the loss is calculated for the preduct value and the ground truth value
    """
    Calculate the cross entropy loss for prediction of one picture
    :param y:
    :param y_pre:
    :return:
    """
    cls_torch = np.full(1, cls)
    pre_cls_torch = y_pre.double()
    target = torch.from_numpy(cls_torch).cuda()
    entropy = nn.CrossEntropyLoss()
    return entropy(pre_cls_torch, target)


def cross_validation_loss(feature_network_path, predict_network_path, src_cls_list, target_path, val_cls_list, class_num, resize_size, crop_size, batch_size, use_gpu):
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

    _, pred_label = predict_network(val_input)

    w = pred_label[0].shape[0]

    error = np.zeros(1)
    error[0] = predict_loss(val_labels[0].item(), pred_label[0].reshape(1, w)).item()
    error = error.reshape(1, 1)
    for num_image in range(1, len(pred_label)):
        single_pred_label = pred_label[num_image]
        w = single_pred_label.shape[0]
        single_val_label = val_labels[num_image]
        error = np.append(error, [[predict_loss(single_val_label.item(), single_pred_label.reshape(1, w)).item()]],
                          axis=0)

    for _ in range(len(iter_val) - 1):
        val_input, val_labels = iter_val.next()
        if use_gpu:
            val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
        else:
            val_input, val_labels = Variable(val_input), Variable(val_labels)
        _, pred_label = predict_network(val_input)
        for num_image in range(len(pred_label)):
            single_pred_label = pred_label[num_image]
            w = single_pred_label.shape[0]
            single_val_label = val_labels[num_image]
            error = np.append(error, [[predict_loss(single_val_label.item(), single_pred_label.reshape(1, w)).item()]],
                              axis=0)

    cross_val_loss = error.sum()
    # for cls in range(class_num):
    #
    #     dsets_val = ImageList(val_cls_list[cls], transform=prep_dict_val)
    #     dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)
    #
    #     # prepare validation feature and predicted label for validation
    #     iter_val = iter(dset_loaders_val)
    #     val_input, val_labels = iter_val.next()
    #     if use_gpu:
    #         val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
    #     else:
    #         val_input, val_labels = Variable(val_input), Variable(val_labels)
    #     val_feature, _ = feature_network(val_input)
    #     _, pred_label = predict_network(val_input)
    #     w, h = pred_label.shape
    #     error = np.zeros(1)
    #     error[0] = predict_loss(cls, pred_label.reshape(1, w*h)).numpy()
    #     error = error.reshape(1,1)
    #     for _ in range(len(val_cls_list[cls]) - 1):
    #         val_input, val_labels = iter_val.next()
    #         # val_feature1 = feature_network(val_input)
    #         val_feature_new, _ = feature_network(val_input)
    #         val_feature = np.append(val_feature, val_feature_new, axis=0)
    #         error = np.append(error, [[predict_loss(cls, predict_network(val_input)[1]).numpy()]], axis=0)
    #
    #     print('The class is {}\n'.format(cls))
    #
    #     cross_val_loss = cross_val_loss + error.sum()
    return cross_val_loss
