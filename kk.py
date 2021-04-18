#!/usr/bin/env Python
# coding=utf-8

import torch
import settings
import numpy as np
import scipy.io as scio
import h5py

if settings.DATASET == "MIRFlickr":
    # set = scio.loadmat(settings.x)
    set_train = scio.loadmat(settings.x_train)
    set_test = scio.loadmat(settings.x_test)

    # label_set = np.array(set['label'], dtype=np.int)  # 21072*38
    # txt_set = np.array(set['text_vec'], dtype=np.float)  # 21072*512
    # img_set = np.array(set['image_fea'], dtype=np.float32)  # 21072*4096

    test_index = np.array(set_test['id_test'], dtype=np.int)  # 1*1000
    test_label_set = np.array(set_test['label_test'], dtype=np.uint8)  # 1000*38
    test_txt_set = np.array(set_test['text_test'], dtype=np.float)  # 1000*512
    test_img_set = np.array(set_test['image_test'], dtype=np.float32)  # 1000*4096

    train_index = np.array(set_train['id_train'], dtype=np.uint8)  # 1*20000
    train_label_set = np.array(set_train['label_train'], dtype=np.int)  # 20000*38
    train_txt_set = np.array(set_train['text_train'], dtype=np.float)  # 20000*512
    train_img_set = np.array(set_train['image_train'], dtype=np.float32)  # 20000*4096

    labeled_train_txt_set = train_txt_set[:1000, :]  # 1000*512
    labeled_train_img_set = train_img_set[:1000, :]  # 1000*4096
    labeled_train_label_set = train_label_set[:1000, :]  # 1000*4096
    labeled_train_index = train_index[:, :1000]

    unlabeled_train_txt_set = train_txt_set[1000:, :]
    unlabeled_train_img_set = train_img_set[1000:, :]
    unlabeled_train_label_set = train_label_set[1000:, :]
    unlabeled_train_index = train_index[:, 1000:]


    # indexTest = test_index.astype(np.uint8)  # ndarray-->1000
    # indexDatabase = train_index.astype(np.uint8)   # ndarray-->20000
    # indexTrain = train_index.astype(np.uint8)   # ndarray-->20000

    class MIRFlickr(torch.utils.data.Dataset):
        def __init__(self, train=True, label=False, unlabel=False, database=False):
            if train:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            elif label:
                self.train_labels = labeled_train_label_set
                self.train_index = labeled_train_index
                self.txt = labeled_train_txt_set
                self.img = labeled_train_img_set
            elif unlabel:
                self.train_labels = unlabeled_train_label_set
                self.train_index = unlabeled_train_index
                self.txt = unlabeled_train_txt_set
                self.img = unlabeled_train_img_set
            elif database:
                self.train_labels = train_label_set
                self.train_index = train_index
                self.txt = train_txt_set
                self.img = train_img_set
            else:
                self.train_labels = test_label_set
                self.train_index = test_index
                self.txt = test_txt_set
                self.img = test_img_set

        def __getitem__(self, index):
            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)


    train_dataset = MIRFlickr(train=True, label=False, unlabel=False, database=False)
    labeled_train_dataset = MIRFlickr(train=False, label=True, unlabel=False, database=False)
    unlabeled_train_dataset = MIRFlickr(train=False, label=False, unlabel=True, database=False)
    test_dataset = MIRFlickr(train=False, label=False, unlabel=False, database=False)
    database_dataset = MIRFlickr(train=False, label=False, unlabel=False, database=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=settings.NUM_WORKERS,
                                           drop_last=False)

labeled_train_loader = torch.utils.data.DataLoader(dataset=labeled_train_dataset,
                                                   batch_size=100,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=False)

unlabeled_train_loader = torch.utils.data.DataLoader(dataset=unlabeled_train_dataset,
                                                     batch_size=20,
                                                     shuffle=True,
                                                     num_workers=settings.NUM_WORKERS,
                                                     drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS,
                                          drop_last=False)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=settings.NUM_WORKERS,
                                              drop_last=False)
