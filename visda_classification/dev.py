import random
import torch
import math
import sys
import psutil
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch.utils.data as util_data
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import seperate_data
from basenet import *


def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error  # weight correspond to Ntr/Nts, error correspond to validation error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    if cov == 0 and var_w == 0:
        cov = var_w = 0.00001
    if var_w == 0:
        var_w = cov # 1/(2.3* cov) or 1.6
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weight(source_feature, target_feature,
               validation_feature):  # 这三个feature根据类别不同，是不一样的. source与target这里需注意一下数据量threshold 2倍的事儿
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    if float(N_s) / N_t > 2:
        source_feature = random_select_src(source_feature, target_feature)
    else:
        source_feature = source_feature.copy()

    print('num_source is {}, num_target is {}, ratio is {}\n'.format(N_s, N_t, float(N_s) / N_t))  # check the ratio

    N_s, d = source_feature.shape
    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)  # 1->source 0->target

    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)

    # here is train, test split, concatenating the data from source and target

    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    domain_classifiers = []

    for decay in decays:
        domain_classifier = MLPClassifier(hidden_layer_sizes=(d, d, 2), activation='relu', alpha=decay)
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
        print('decay is %s, val acc is %s' % (decay, acc))

    index = val_acc.index(max(val_acc))

    # print('val acc is')
    # print(val_acc)

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t  # (Ntr/Nts)*(1-M(fv))/M(fv)

    # correspond to (Ntr/Nts)*(1-M(fv))/M(fv), M(fv) just indicate whether 0 or 1, meaning from source or target

def random_select_src(source_feature, target_feature):
    # done with debugging
    """
    Select at most 2*Ntr data from source feature randomly
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    items = [i for i in range(1, N_s)]
    random_list = random.sample(items, 2 * N_t - 1)
    new_source_feature = source_feature[0].reshape(1, d)
    for i in range(2 * N_t - 1):
        new_source_feature = np.concatenate((new_source_feature, source_feature[random_list[i]].reshape(1, d)))

    print("random_select:")
    print(new_source_feature.shape)
    return new_source_feature


def predict_loss(cls, y_pre):
    # done with debugging works fine
    """
    Calculate the cross entropy loss for prediction of one picture
    :param cls:
    :param y_pre:
    :return:
    """
    cls_torch = np.full(1, cls)
    pre_cls_torch = y_pre.double()
    target = torch.from_numpy(cls_torch).cuda()
    entropy = nn.CrossEntropyLoss()
    return entropy(pre_cls_torch, target)

def get_label_list(args, target_list, feature_network_path, predict_network_path, num_layer, resize_size, crop_size, batch_size, use_gpu):
    """
    Return the target list with pesudolabel
    :param target_list: list conatinging all target file path and a wrong label
    :param predict_network: network to perdict label for target image
    :param resize_size:
    :param crop_size:
    :param batch_size:
    :return:
    """
    option = 'resnet' + args.resnet
    G = ResBase(option)
    F1 = ResClassifier(num_layer=num_layer)

    G.load_state_dict(torch.load(feature_network_path))
    F1.load_state_dict(torch.load(predict_network_path))
    if use_gpu:
        G.cuda()
        F1.cuda()
    G.eval()
    F1.eval()

    label_list = []
    dsets_tar = ImageList(target_list, transform=prep.image_train(resize_size=resize_size, crop_size=crop_size))
    dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=False, num_workers=4)
    len_train_target = len(dset_loaders_tar)
    iter_target = iter(dset_loaders_tar)
    count = 0
    for i in range(len_train_target):
        input_tar, label_tar = iter_target.next()
        if use_gpu:
            input_tar, label_tar = Variable(input_tar).cuda(), Variable(label_tar).cuda()
        else:
            input_tar, label_tar = Variable(input_tar), Variable(label_tar)
        tar_feature = G(input_tar)
        predict_score = F1(tar_feature)
        _, pre_lab = torch.max(predict_score, 1)
        predict_label = pre_lab.detach()
        for num in range(len(predict_label.cpu())):
            if target_list[count][-3] == ' ':
                ind = -2
            else:
                ind = -3
            label_list.append(target_list[count][:ind])
            label_list[count] = label_list[count] + str(predict_label[num].cpu().numpy()) + "\n"
            count += 1
    return label_list


def cross_validation_loss(args, feature_network_path, predict_network_path, num_layer, src_cls_list, target_path, val_cls_list, class_num,
                          resize_size, crop_size, batch_size, use_gpu):
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
    target_list_no_label = open(target_path).readlines()
    tar_cls_list = []
    cross_val_loss = 0

    # add pesudolabel for target data
    target_list = get_label_list(args, target_list_no_label, feature_network_path, predict_network_path, num_layer,
                                 resize_size, crop_size, batch_size, use_gpu)

    # seperate the class
    for i in range(class_num):
        tar_cls_list.append([j for j in target_list if int(j.split(" ")[1].replace("\n", "")) == i])
    prep_dict = prep.image_train(resize_size=resize_size, crop_size=crop_size)
    # load network
    option = 'resnet' + args.resnet
    G = ResBase(option)
    F1 = ResClassifier(num_layer=num_layer)
    G.load_state_dict(torch.load(feature_network_path))
    F1.load_state_dict(torch.load(predict_network_path))
    if use_gpu:
        G.cuda()
        F1.cuda()
    G.eval()
    F1.eval()
    # load different class's image
    for cls in range(class_num):
        print(cls)
        # prepare source feature
        dsets_src = ImageList(src_cls_list[cls], transform=prep_dict)
        dset_loaders_src = util_data.DataLoader(dsets_src, batch_size=batch_size, shuffle=True, num_workers=4)

        iter_src = iter(dset_loaders_src)
        src_input = iter_src.next()[0]
        if use_gpu:
            src_input = Variable(src_input).cuda()
        else:
            src_input = Variable(src_input)

        src_feature = G(src_input)
        src_feature_de = src_feature.detach().cpu().numpy()
        for count_src in range(len(dset_loaders_src) - 1):
            src_input = iter_src.next()[0]
            if use_gpu:
                src_input = Variable(src_input).cuda()
            else:
                src_input = Variable(src_input)

            src_feature_new = G(src_input)
            src_feature_new_de = src_feature_new.detach().cpu().numpy()
            src_feature_de = np.append(src_feature_de, src_feature_new_de, axis=0)

        # prepare target feature
        dsets_tar = ImageList(tar_cls_list[cls], transform=prep_dict)
        dset_loaders_tar = util_data.DataLoader(dsets_tar, batch_size=batch_size, shuffle=True, num_workers=4)

        iter_tar = iter(dset_loaders_tar)
        tar_input = iter_tar.next()[0]
        if use_gpu:
            tar_input = Variable(tar_input).cuda()
        else:
            tar_input = Variable(tar_input)

        tar_feature = G(tar_input)
        tar_feature_de = tar_feature.detach().cpu().numpy()
        for count_tar in range(len(dset_loaders_tar) - 1):
            tar_input = iter_tar.next()[0]
            if use_gpu:
                tar_input = Variable(tar_input).cuda()
            else:
                tar_input = Variable(tar_input)

            tar_feature_new = G(tar_input)
            tar_feature_new_de = tar_feature_new.detach().cpu().numpy()
            tar_feature_de = np.append(tar_feature_de, tar_feature_new_de, axis=0)

        # prepare validation feature and predicted label for validation
        dsets_val = ImageList(val_cls_list[cls], transform=prep_dict)
        dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)

        iter_val = iter(dset_loaders_val)
        val_input, val_labels = iter_val.next()
        if use_gpu:
            val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
        else:
            val_input, val_labels = Variable(val_input), Variable(val_labels)

        val_feature = G(val_input)
        pred_label = F1(val_feature)
        val_feature_de = val_feature.detach().cpu().numpy()

        w = pred_label[0].shape[0]
        error = np.zeros(1)

        error[0] = predict_loss(cls, pred_label[0].reshape(1, w)).item()
        error = error.reshape(1, 1)
        for num_image in range(1, len(pred_label)):
            new_error = np.zeros(1)
            single_pred_label = pred_label[num_image]
            w = single_pred_label.shape[0]

            new_error[0] = predict_loss(cls, single_pred_label.reshape(1, w)).item()
            new_error = new_error.reshape(1, 1)
            error = np.append(error, new_error, axis=0)
        for count_val in range(len(dset_loaders_val) - 1):
            val_input, val_labels = iter_val.next()
            if use_gpu:
                val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
            else:
                val_input, val_labels = Variable(val_input), Variable(val_labels)

            val_feature_new = G(val_input)
            val_feature_new_de = val_feature_new.detach().cpu().numpy()
            val_feature_de = np.append(val_feature_de, val_feature_new_de, axis=0)
            val_feature = G(val_input)
            pred_label = F1(val_feature)

            for num_image in range(len(pred_label)):
                new_error = np.zeros(1)
                single_pred_label = pred_label[num_image]
                w = single_pred_label.shape[0]
                new_error[0] = predict_loss(cls, single_pred_label.reshape(1, w)).item()
                new_error = new_error.reshape(1, 1)
            # cls should be a value, new_labels should be a [[x]] tensor format, the input format required by predict_loss
                error = np.append(error, new_error, axis=0)
            # error should be a (N, 1) numpy array, the input format required by get_dev_risk
        print(src_feature_de.shape)
        print(tar_feature_de.shape)
        print(val_feature_de.shape)

        # print(cls)
        weight = get_weight(src_feature_de, tar_feature_de, val_feature_de)
        cross_val_loss = cross_val_loss + get_dev_risk(weight, error)/class_num


    return cross_val_loss
