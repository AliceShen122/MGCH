#!/usr/bin/env Python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, cluster_acc, target_distribution, precision_recall, \
    precision_top_k, optimized_mAP
import kk
import settings
from models import TxtNet, ImgNet
import os.path as osp
from similarity_matrix import similarity_matrix
import io
import os
from GCN import GCN
from tqdm import tqdm
from itertools import cycle
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from resnet import ResNet
from MSText import get_MS_Text


class Session:
    def __init__(self):
        self.logger = settings.logger
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.gcn = GCN(4608, settings.CODE_LEN, 38, 120)
        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN)
        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                                     weight_decay=settings.WEIGHT_DECAY)
        self.opt_gcn = torch.optim.Adam(self.gcn.parameters(), lr=0.001
                                        # , momentum=settings.MOMENTUM,weight_decay=settings.WEIGHT_DECAY
                                        )
        self.logSoftloss = torch.nn.BCEWithLogitsLoss()
        self.MSELoss = torch.nn.MSELoss()
        self.similarity_matrix = similarity_matrix()
        # self.img_buffer = torch.randn(10000, 4096).cuda()
        # self.txt_buffer = torch.randn(10000, 4096).cuda()

    def train(self):
        self.CodeNet_I.train().cuda()
        self.CodeNet_T.train().cuda()
        self.gcn.train().cuda()

        anchor_image_1 = kk.labeled_train_img_set[:800, :]  # (800,4096)
        anchor_text_1 = kk.labeled_train_txt_set[:800, :]  # (800,512)
        anchor_label_1 = kk.labeled_train_label_set[:800, :]  # (800,80)

        # anchor_image_2 = kk.labeled_train_img_set[800:1600, :]  # (800,4096)
        # anchor_text_2 = kk.labeled_train_txt_set[800:1600, :]  # (800,512)
        # anchor_label_2 = kk.labeled_train_label_set[800:1600, :]  # (800,80)

        # anchor_image_3 = kk.labeled_train_img_set[1600:2400, :]  # (800,4096)
        # anchor_text_3 = kk.labeled_train_txt_set[1600:2400, :]  # (800,512)
        # anchor_label_3 = kk.labeled_train_label_set[1600:2400, :]  # (800,80)

        # anchor_image_4 = kk.labeled_train_img_set[2400:3200, :]  # (800,4096)
        # anchor_text_4 = kk.labeled_train_txt_set[2400:3200, :]  # (800,512)
        # anchor_label_4 = kk.labeled_train_label_set[2400:3200, :]  # (800,80)

        # anchor_image_5 = kk.labeled_train_img_set[2400:3200, :]  # (800,4096)
        # anchor_text_5 = kk.labeled_train_txt_set[2400:3200, :]  # (800,512)
        # anchor_label_5 = kk.labeled_train_label_set[2400:3200, :]  # (800,80)

        # anchor_image_6 = kk.labeled_train_img_set[3200:3600, :]  # (800,4096)
        # anchor_text_6 = kk.labeled_train_txt_set[3200:3600, :]  # (800,512)
        # anchor_label_6 = kk.labeled_train_label_set[3200:3600, :]  # (800,80)

        # anchor_label = torch.cat((anchor_label, anchor_label), dim=1)  # (1600,80)
        # anchor_w = anchor_image.mm(anchor_text.t())  # torch.norm (800,800)
        # anchor_w = torch.nn.functional.softmax(anchor_w, dim=1)  # 按行计算
        # anchor_fused = 1 / 2 * (anchor_w.mm(anchor_image) + anchor_w.mm(anchor_text))  # (800,4096)
        # anchor_fused = zero_mean(anchor_fused)

        anchor_fused_1 = torch.cat((torch.Tensor(anchor_image_1), torch.Tensor(anchor_text_1)), dim=1)  # (800,4608)
        anchor_affnty_1, _ = labels_affnty(anchor_label_1, anchor_label_1)  # (800,800)
        anchor_affnty_1 = Variable(anchor_affnty_1).cuda()  # (800,800)

        # anchor_fused_2 = torch.cat((torch.Tensor(anchor_image_2), torch.Tensor(anchor_text_2)), dim=1)  # (800,4608)
        # anchor_affnty_2, _ = labels_affnty(anchor_label_2, anchor_label_2)  # (800,800)
        # anchor_affnty_2 = Variable(anchor_affnty_2).cuda()  # (800,800)

        # anchor_fused_3 = torch.cat((torch.Tensor(anchor_image_3), torch.Tensor(anchor_text_3)), dim=1)  # (800,4608)
        # anchor_affnty_3, _ = labels_affnty(anchor_label_3, anchor_label_3)  # (800,800)
        # anchor_affnty_3 = Variable(anchor_affnty_3).cuda()  # (800,800)

        # anchor_fused_4 = torch.cat((torch.Tensor(anchor_image_4), torch.Tensor(anchor_text_4)), dim=1)  # (800,4608)
        # anchor_affnty_4, _ = labels_affnty(anchor_label_4, anchor_label_4)  # (800,800)
        # anchor_affnty_4 = Variable(anchor_affnty_4).cuda()  # (800,800)

        # anchor_fused_5 = torch.cat((torch.Tensor(anchor_image_5), torch.Tensor(anchor_text_5)), dim=1)  # (800,4608)
        # anchor_affnty_5, _ = labels_affnty(anchor_label_5, anchor_label_5)  # (800,800)
        # anchor_affnty_5 = Variable(anchor_affnty_5).cuda()  # (800,800)

        # anchor_fused_6 = torch.cat((torch.Tensor(anchor_image_6), torch.Tensor(anchor_text_6)), dim=1)  # (800,4608)
        # anchor_affnty_6, _ = labels_affnty(anchor_label_6, anchor_label_6)  # (800,800)
        # anchor_affnty_6 = Variable(anchor_affnty_6).cuda()  # (800,800)

        labeled_train_loader = cycle(kk.labeled_train_loader)
        for epoch in range(80):
            # i = 0
            self.CodeNet_I.set_alpha(epoch)
            self.CodeNet_T.set_alpha(epoch)
            for data in tqdm(kk.unlabeled_train_loader):  # 0-image 1-txt 2-label 3-index  (10000/40=250)
                # i += 1
                # index = data[3].numpy()  # ndarray->(64,)
                labeled_data = next(labeled_train_loader)  #
                l_img = labeled_data[0]  # (40,4096)
                l_txt = labeled_data[1]  # (40,512)
                l_label = labeled_data[2]  # (40,80)
                l_img = l_img.cuda()
                l_txt = l_txt.to(torch.float32).cuda()

                un_img = data[0]  # type: torch.Tensor (80,4096)
                un_txt = data[1]  # type: torch.Tensor  (80,512)
                un_img = un_img.cuda()  # float32
                un_txt = un_txt.to(torch.float32).cuda()  # float32

                labeled_fused = torch.cat((l_img, l_txt), dim=1)  # (40,4608)
                unlabeled_fused = torch.cat((un_img, un_txt), dim=1)  # (80,4608)

                feature_fused = Variable(torch.cat((labeled_fused, unlabeled_fused), 0))  # (120,4608)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                in_aff_1, out_aff_1 = get_affnty(l_label, anchor_label_1, unlabeled_fused, anchor_fused_1,
                                                 topk=3)  # in_aff=(800,120) out_aff = (120,800)
                # in_aff_2, out_aff_2 = get_affnty(l_label, anchor_label_2, unlabeled_fused, anchor_fused_2,
                #                                  topk=3)  # in_aff=(800,120) out_aff = (120,800)
                # in_aff_3, out_aff_3 = get_affnty(l_label, anchor_label_3, unlabeled_fused, anchor_fused_3,
                #                                  topk=3)  # in_aff=(1600,240) out_aff = (240,1600)
                # in_aff_4, out_aff_4 = get_affnty(l_label, anchor_label_4, unlabeled_fused, anchor_fused_4,
                #                                  topk=3)  # in_aff=(1600,240) out_aff = (240,1600)
                # in_aff_5, out_aff_5 = get_affnty(l_label, anchor_label_5, unlabeled_fused, anchor_fused_5,
                #                                  topk=3)  # in_aff=(1600,240) out_aff = (240,1600)
                # in_aff_6, out_aff_6 = get_affnty(l_label, anchor_label_6, unlabeled_fused, anchor_fused_6,
                #                                  topk=3)  # in_aff=(1600,240) out_aff = (240,1600)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                l_label = Variable(torch.LongTensor(l_label.numpy())).cuda()  # (40,80)

                self.opt_gcn.zero_grad()
                self.opt_I.zero_grad()
                self.opt_T.zero_grad()

                un_img_v, un_hid_i, un_hash_i = self.CodeNet_I(un_img)  # type: torch.Tensor (80,4096) (80,64) (80,64)
                un_txt_v, un_hid_t, un_hash_t = self.CodeNet_T(un_txt)  # type: torch.Tensor (80,4096) (80,64) (80,64)
                l_img_v, l_hid_i, l_hash_i = self.CodeNet_I(l_img)  # type: torch.Tensor (40,4096) (40,64) (40,64)
                l_txt_v, l_hid_t, l_hash_t = self.CodeNet_T(l_txt)  # type: torch.Tensor (40,4096) (40,64) (40,64)

                hash_i = torch.cat((l_hash_i, un_hash_i), dim=0)  # (120,64)
                hash_t = torch.cat((l_hash_t, un_hash_t), dim=0)  # (120,64)
                hash_i = hash_i.cuda()
                hash_t = hash_t.cuda()

                outputs, pred = self.gcn(feature_fused, in_aff_1, out_aff_1, anchor_affnty_1,
                                         labeled_fused.size(0))  # (120,64) (40,80)
                # outputs, pred = self.gcn(feature_fused, in_aff_1, out_aff_1, anchor_affnty_1,
                #                          labeled_fused.size(0))  # (120,64) (40,80)
                # pred = torch.softmax(pred, dim=1)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                B = torch.sign(outputs).data.cpu().numpy()  # (120,64)
                binary_target = Variable(torch.Tensor(B)).cuda()  # (120,64)
                quanloss = self.MSELoss(outputs, binary_target) * 0.0001
                # normloss = torch.norm(torch.dot(outputs.t(), outputs), 2) * settings.Lambda
                sumloss = torch.sum(outputs) / outputs.size(0) * settings.mu
                mseloss1 = self.MSELoss(hash_i, binary_target) * settings.alpha
                mseloss2 = self.MSELoss(hash_t, binary_target) * settings.beta
                logSoftloss = self.logSoftloss(pred, l_label.float())  # 加sigmod

                # if epoch_before + 1 == 10:
                #     if i == 1:
                #         img_vector = img_v
                #         txt_vector = txt_v
                #         label_vector = label
                #     else:
                #         img_vector = torch.cat((img_vector, img_v), 0)  # (10000,4096)
                #         txt_vector = torch.cat((txt_vector, txt_v), 0)  # (10000,4096)
                #         label_vector = torch.cat((label_vector, label), 0)  # (10000,80)
                # self.img_buffer[index, :] = image.data
                # img_buffer = Variable(self.img_buffer)  # float32 (10000,4096)
                # self.txt_buffer[index, :] = txt_v.data
                # txt_buffer = Variable(self.txt_buffer)  # float32 (10000,4096)

                # l = torch.Tensor(kk.labeled_train_label_set).float().cuda()
                # S = calc_neighbor(label, l)  # (64,10000)
                # theta1 = 1.0 / 2 * torch.matmul(txt_v, img_buffer.t())  # (64,4096)(4096,10000)->(64,10000)
                # loss1 = -torch.mean(S * theta1 - torch.log(1 + torch.exp(theta1)))
                # theta2 = 1.0 / 2 * torch.matmul(image, txt_buffer.t())  # (64,4096)(4096,10000)->(64,10000)
                # loss2 = -torch.mean(S * theta2 - torch.log(1 + torch.exp(theta2)))
                # loss_before = loss1 + loss2

                loss = logSoftloss + quanloss + sumloss + mseloss1 + mseloss2  # + NORMloss NORMloss, + SUMloss SUMloss, + loss1, loss1.item(), loss1: %.4f
                loss.backward()
                self.opt_T.step()
                self.opt_I.step()
                self.opt_gcn.step()
                self.logger.info(
                    'Epoch [%d/%d], logSoftloss: %.4f, quanloss: %.4f, mseloss1: %.4f, mseloss2: %.4f, Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH,
                       logSoftloss.item(), quanloss.item(), mseloss1.item(), mseloss2.item(), loss.item()))

            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
                self.CodeNet_I.eval().cuda()
                self.CodeNet_T.eval().cuda()
                # self.gcn.eval().cuda()
                re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(kk.database_loader, kk.test_loader, self.CodeNet_I,
                                                                  self.CodeNet_T)
                MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
                MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
                i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, self.similarity_matrix)
                t2i_pre, t2i_recall = precision_recall(qu_BT, re_BI, self.similarity_matrix)
                with io.open('results_%s' % settings.DATASET + '/results_%d.txt' % settings.CODE_LEN, 'a',
                             encoding='utf-8') as f:
                    f.write(u'MAP_I2T: ' + str(MAP_I2T) + '\n')
                    f.write(u'MAP_T2I: ' + str(MAP_T2I) + '\n')
                    f.write(u'i2t precision: ' + str(i2t_pre) + '\n')
                    f.write(u'i2t recall: ' + str(i2t_recall) + '\n')
                    f.write(u't2i precision: ' + str(t2i_pre) + '\n')
                    f.write(u't2i recall: ' + str(t2i_recall) + '\n\n')
                self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
                self.logger.info('--------------------------------------------------------------------')

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_T.eval().cuda()
        self.gcn.eval().cuda()
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(kk.database_loader, kk.test_loader, self.CodeNet_T,
                                                          self.gcn, kk.database_dataset, kk.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        # i2t_map = optimized_mAP(qu_BI, re_BT, self.similarity_matrix, 'hash',
        #                         top=50000)
        # i2t_pre_top_k = precision_top_k(qu_BI, re_BT, self.similarity_matrix,
        #                                 [1, 500, 1000, 1500, 2000], 'hash')
        # i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, self.similarity_matrix)

        # t2i_map = optimized_mAP(qu_BT, re_BI, self.similarity_matrix, 'hash', top=50000)
        # t2i_pre_top_k = precision_top_k(qu_BT, re_BI, self.similarity_matrix,
        #                                 [1, 500, 1000, 1500, 2000], 'hash')
        # t2i_pre, t2i_recall = precision_recall(qu_BT, re_BI, self.similarity_matrix)

        with io.open('results_%s' % settings.DATASET + '/results_%d.txt' % settings.CODE_LEN, 'a',
                     encoding='utf-8') as f:
            f.write(u'MAP_I2T: ' + str(MAP_I2T) + '\n')
            f.write(u'MAP_T2I: ' + str(MAP_T2I) + '\n')
        #     f.write(u'i2t_map: ' + str(i2t_map) + '\n')
        #     f.write(u't2i_map: ' + str(t2i_map) + '\n')
        #     f.write(u'i2t_pre_top_k: ' + str(i2t_pre_top_k) + '\n')
        #     f.write(u't2i_pre_top_k: ' + str(t2i_pre_top_k) + '\n')
        #     f.write(u'i2t precision: ' + str(i2t_pre) + '\n')
        #     f.write(u'i2t recall: ' + str(i2t_recall) + '\n')
        #     f.write(u't2i precision: ' + str(t2i_pre) + '\n')
        #     f.write(u't2i recall: ' + str(t2i_recall) + '\n\n')

        # self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (i2t_map, t2i_map))
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')

    # def save_checkpoints(self, step, file_name='latest.pth'):
    #     ckp_path = osp.join(settings.MODEL_DIR, file_name)
    #     obj = {
    #         'ImgNet': self.CodeNet_I.state_dict(),
    #         'TxtNet': self.CodeNet_T.state_dict(),
    #         'step': step,
    #     }
    #     torch.save(obj, ckp_path)
    #     self.logger.info('**********Save the trained model successfully.**********')

    # def load_checkpoints(self, file_name='latest.pth'):
    #     ckp_path = osp.join(settings.MODEL_DIR, file_name)
    #     try:
    #         obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
    #         self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
    #     except IOError:
    #         self.logger.error('********** No checkpoint %s!*********' % ckp_path)
    #         return
    #     self.CodeNet_I.load_state_dict(obj['ImgNet'])
    #     self.CodeNet_T.load_state_dict(obj['TxtNet'])
    #     self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


# construct affinity matrix via supervised labels information
def labels_affnty(labels_1, labels_2):  # 40,80;800,80 anhcor_label:(800,38)
    if (isinstance(labels_1, torch.LongTensor) or
            isinstance(labels_1, torch.Tensor)):
        labels_1 = labels_1.cpu().numpy()
    if (isinstance(labels_2, torch.LongTensor) or
            isinstance(labels_2, torch.Tensor)):
        labels_2 = labels_2.cpu().numpy()

    if labels_1.ndim == 1:  # 数组的维度
        affnty = np.float32(labels_2 == labels_1[:, np.newaxis])  # labels_1: (800,)->(800,1)
        # np.newaxis的作用是增加一个维度。对于[: , np.newaxis] 和 [np.newaxis，：],是在np.newaxis这里增加1维。这样改变维度的作用往往是将一维的数据转变成一个矩阵
    else:
        affnty = np.float32(np.sign(np.dot(labels_1, labels_2.T)))
    in_affnty, out_affnty = normalize(affnty)
    return torch.Tensor(in_affnty), torch.Tensor(out_affnty)  # (800,800) (800,800)


def zero2eps(x):
    x[x == 0] = 1
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = torch.mean(x, 0)
    x -= mean_val
    return x


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


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


def get_affnty(labels1, labels2, X, Y, topk=10):
    # l_label(40,80),anchor_label(800,80),unlabel_fused(80,4608),anchor_fused(800,4608),topk=3
    in_affnty1, out_affnty1 = labels_affnty(labels1, labels2)  # (800,40) (40,800)
    in_affnty2, out_affnty2 = rbf_affnty(X, Y, topk)  # (800,80) (80,800)

    in_affnty = torch.cat((in_affnty1, in_affnty2), 1)  # (800,120)
    out_affnty = torch.cat((out_affnty1, out_affnty2), 0)  # (120,800)
    return Variable(in_affnty).cuda(), Variable(out_affnty).cuda()


def main():
    sess = Session()

    if settings.EVAL == True:
        # sess.load_checkpoints()
        sess.eval()

    else:
        # for epoch in range(settings.NUM_EPOCH):
        # train the Model
        sess.train()
        # eval the Model
        # if (epoch + 1) % settings.EVAL_INTERVAL == 0:
        #     sess.eval()
        # save the model
        # if epoch + 1 == settings.NUM_EPOCH:
        #     sess.save_checkpoints(step=epoch + 1)


if __name__ == '__main__':
    main()
