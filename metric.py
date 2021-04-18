#!/usr/bin/env Python
# coding=utf-8

from __future__ import division  # 用于/相除的时候,保留真实结果.小数

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import scipy.io as sio
import numpy as np
import settings
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel
import kk


def compress(train_loader, test_loader, model_I, model_T):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, _, code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, _, code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = kk.train_label_set

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = kk.test_label_set
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


# def compress(train_loader, test_loader, model_T, gcn, anchor_fused, anchor_affnty):
#     re_BI = list([])
#     re_BT = list([])
#     for _, (data_I, data_T, _, _) in enumerate(train_loader):
#         in_aff, out_aff = rbf_affnty(data_I, anchor_fused, topk=20)  # (800,64) (64,800)
#         var_data_I = Variable(data_I.cuda())
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(var_data_I, in_aff, out_aff, anchor_affnty, 1)  # (64,64)
#         code_I = torch.sign(out)
#         re_BI.extend(code_I.cpu().data.numpy())
#
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         feat_T,_ = model_T(var_data_T)
#         # w = var_data_I.mm(feat_T.t())  # (80,80)
#         # w = torch.nn.functional.softmax(w, dim=1)
#         # fused = 1 / 2 * (w.mm(var_data_I) + w.mm(feat_T))  # (80,512)
#         in_aff, out_aff = rbf_affnty(feat_T, anchor_fused, topk=20)  # (800,64) (64,800)
#         feat_T = Variable(feat_T).cuda()
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(feat_T, in_aff, out_aff, anchor_affnty, 1)  # (64,64)
#         code_T = torch.sign(out)
#         re_BT.extend(code_T.cpu().data.numpy())
#     qu_BI = list([])
#     qu_BT = list([])
#     for _, (data_I, data_T, _, _) in enumerate(test_loader):
#         in_aff, out_aff = rbf_affnty(data_I, anchor_fused, topk=20)  # (800,64) (64,800)
#         var_data_I = Variable(data_I.cuda())
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(var_data_I, in_aff, out_aff, anchor_affnty, 1)  # (64,64)
#         code_I = torch.sign(out)
#         qu_BI.extend(code_I.cpu().data.numpy())
#
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         feat_T,_ = model_T(var_data_T)
#         # w = var_data_I.mm(feat_T.t())  # (80,80)
#         # w = torch.nn.functional.softmax(w, dim=1)
#         # fused = 1 / 2 * (w.mm(var_data_I) + w.mm(feat_T))  # (80,512)
#         in_aff, out_aff = rbf_affnty(feat_T, anchor_fused, topk=20)  # (800,64) (64,800)
#         feat_T = Variable(feat_T).cuda()
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(feat_T, in_aff, out_aff, anchor_affnty, 1)  # (64,64)
#         code_T = torch.sign(out)
#         qu_BT.extend(code_T.cpu().data.numpy())
#     re_BI = np.array(re_BI, dtype=np.double)
#     re_BT = np.array(re_BT, dtype=np.double)
#     re_L = kk.train_label_set
#     qu_BI = np.array(qu_BI, dtype=np.double)
#     qu_BT = np.array(qu_BT, dtype=np.double)
#     qu_L = kk.test_label_set
#     sio.savemat('codes_%s' % settings.DATASET + '/codes_%d.mat' % settings.CODE_LEN,
#                 {'re_BI': re_BI, 're_BT': re_BT, 're_L': re_L, 'qu_BI': qu_BI, 'qu_BT': qu_BT, 'qu_L': qu_L})
#     return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

# def compress(train_loader, test_loader, gcn, anchor_fused, anchor_affnty):
#     re_BI = list([])
#     re_BT = list([])
#     for _, (data_I, data_T, _, _) in enumerate(train_loader):
#         var_data_I = Variable(data_I.cuda())  # 64,4096
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())  # 64,512
#         # feat_I,_ = model_I(var_data_I)  # (64,4096)
#         # feat_T,_ = model_T(var_data_T)  # (64,4096)
#         fused = torch.cat((var_data_I, var_data_T), dim=1)  # (128,4096)
#         in_aff, out_aff = rbf_affnty(fused, anchor_fused, topk=20)  # anchor_fused=(800,4608)
#         fused = Variable(fused).cuda()  # (128,4096)
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(fused, in_aff, out_aff, anchor_affnty, 1)  # (128,64)
#         code_I = torch.sign(out)
#         re_BI.extend(code_I.cpu().data.numpy())
#         code_T = torch.sign(out)
#         re_BT.extend(code_T.cpu().data.numpy())
#     qu_BI = list([])
#     qu_BT = list([])
#     for _, (data_I, data_T, _, _) in enumerate(test_loader):
#         var_data_I = Variable(data_I.cuda())
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         # feat_I,_ = model_I(var_data_I)
#         # feat_T,_ = model_T(var_data_T)
#         fused = torch.cat((var_data_I, var_data_T), dim=1)  # (64,4608)
#         in_aff, out_aff = rbf_affnty(fused, anchor_fused, topk=20)  # (800,64) (64,800)
#         fused = Variable(fused).cuda()
#         in_aff = Variable(in_aff).cuda()
#         out_aff = Variable(out_aff).cuda()
#         out = gcn(fused, in_aff, out_aff, anchor_affnty, 1)  # (64,64)
#         code_I = torch.sign(out)
#         qu_BI.extend(code_I.cpu().data.numpy())
#         code_T = torch.sign(out)
#         qu_BT.extend(code_T.cpu().data.numpy())
#     re_BI = np.array(re_BI, dtype=np.double)
#     re_BT = np.array(re_BT, dtype=np.double)
#     re_L = kk.train_label_set
#     qu_BI = np.array(qu_BI, dtype=np.double)
#     qu_BT = np.array(qu_BT, dtype=np.double)
#     qu_L = kk.test_label_set
#     sio.savemat('codes_%s' % settings.DATASET + '/codes_%d.mat' % settings.CODE_LEN,
#                 {'re_BI': re_BI, 're_BT': re_BT, 're_L': re_L, 'qu_BI': qu_BI, 'qu_BT': qu_BT, 'qu_L': qu_L})
#     return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affnty):  # affnty(800,800)
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])  # axis=1: ndarray的每一行相加->(800,)->(800,1)
    row_sum = zero2eps(np.sum(affnty, axis=0))  # axis=0: ndarray的每一列相加->(800,)

    out_affnty = affnty / col_sum  # (800,800)
    in_affnty = np.transpose(affnty / row_sum)  # (800,800)
    return in_affnty, out_affnty


# construct affinity matrix via rbf kernel
def rbf_affnty(X, Y, topk=10):
    X = X.cpu().detach().numpy()  # (40,512)
    Y = Y.cpu().detach().numpy()  # (800,512)

    rbf_k = rbf_kernel(X, Y)  # 高斯核 (40,800)
    topk_max = np.argsort(rbf_k, axis=1)[:, -topk:]  # (40,3) 每一行的数进行比较,选取最大的topk个; argsort()是将X中的元素从小到大排序后,提取对应的索引index

    affnty = np.zeros(rbf_k.shape)  # (40,800)
    for col_idx in topk_max.T:  # (3,40); topk_max的第一列,第二列,第三列,每一列40个数
        affnty[np.arange(rbf_k.shape[0]), col_idx] = 1.0  # np.arange(rbf_k.shape[0])=0,1,2,3,4,...,39

    in_affnty, out_affnty = normalize(affnty)  # (800,40) (40,800)
    return torch.Tensor(in_affnty), torch.Tensor(out_affnty)

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def precision_recall(q, r, similarity_matrix):
    pre_list = []
    recall_list = []
    query = q.copy()  # (2000,16)
    retrieval = r.copy()  # (18015,16)

    query[query >= 0] = 1
    query[query != 1] = 0
    retrieval[retrieval >= 0] = 1
    retrieval[retrieval != 1] = 0  # 将-1变为0

    query = query.astype(dtype=np.int8)
    retrieval = retrieval.astype(dtype=np.int8)

    distance = hamming_distance(query, retrieval)  # (2000,18015)
    distance_max = np.max(distance)  # 15
    distance_min = np.min(distance)  # 4

    for radius in range(int(distance_min), int(distance_max)):
        temp_distance = distance.copy()

        temp_distance[distance <= radius] = 1
        temp_distance[temp_distance > radius] = 0

        tp = np.sum(similarity_matrix * temp_distance)  # 7790,67581,723031
        precision = 0
        recall = 0
        if tp != 0:
            x = np.sum(temp_distance)  # 109
            y = np.sum(similarity_matrix) # 20848629
            precision = tp / x
            recall = tp / y
        pre_list.append(precision)
        recall_list.append(recall)

    pre_list = [round(i, 4) for i in pre_list]
    recall_list = [round(i, 4) for i in recall_list]

    return pre_list, recall_list


def hamming_distance(X, Y):
    '''
    返回两个矩阵以行为pair的汉明距离
    :param X: (n, hash_len)
    :param Y: (m, hash_len)
    :return: (n, m)
    '''

    res = np.bitwise_xor(np.expand_dims(X, 1), np.expand_dims(Y, 0))
    res = np.sum(res, axis=2)
    return res


def optimized_mAP(q, r, similarity_matrix, dis_metric='hash', top=None):
    query = q.copy()
    retrieval = r.copy()

    query_size = query.shape[0]

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = hamming_distance(query, retrieval)
    elif dis_metric == 'eu':
        distance = euclidean_distances(query, retrieval)
    else:
        distance = cosine_similarity(query, retrieval)

    sorted_index = np.argsort(distance)
    if dis_metric == 'cosine':
        sorted_index = np.flip(sorted_index, axis=1)

    sorted_similarity_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    sorted_similarity_matrix = np.asarray(sorted_similarity_matrix)[:, :top]
    neighbors = np.sum(sorted_similarity_matrix, axis=1)
    one_index = np.argwhere(sorted_similarity_matrix == 1)
    precision = 0
    cnt = 0

    for i in range(query_size):
        precision_at_i = 0
        if neighbors[i] == 0:
            continue
        for j in range(neighbors[i]):
            precision_at_i += np.sum(sorted_similarity_matrix[i, :one_index[cnt, 1] + 1]) / (one_index[cnt, 1] + 1)
            cnt += 1
        precision += precision_at_i / neighbors[i]
    mAP = precision / query_size

    return mAP


def precision_top_k(q, r, similarity_matrix, top_k, dis_metric):
    # calculate top k precision
    query = q.copy()
    retrieval = r.copy()

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = calculate_hamming(query, retrieval)

    sorted_index = np.argsort(distance)

    sorted_simi_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    precision = []
    for i in top_k:
        average_precison_top_i = np.mean(np.sum(sorted_simi_matrix[:, :i], axis=1) / i)
        precision.append(average_precison_top_i)

    precision = [round(i, 4) for i in precision]
    return precision