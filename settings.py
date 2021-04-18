#!/usr/bin/env Python
# coding=utf-8

import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

# dataset can be 'MIRFlickr' or 'NUSWIDE'
DATASET = 'MIRFlickr'

if DATASET == 'MIRFlickr':
    x = '/data/sx/CrossModal/mir.mat'
    x_train = '/data/sx/CrossModal/mir_train.mat'
    x_test = '/data/sx/CrossModal/mir_test.mat'
    
    # BETA = 0.9
    # LAMBDA1 = 0.1
    # LAMBDA2 = 0.1
    NUM_EPOCH = 100
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 2

BATCH_SIZE = 64
CODE_LEN = 64
alpha = 1
beta = 1
Lambda = 0.0001
mu = 0.00001
# MU = 1.5
# ETA = 0.4


MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)


logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
# logger.info('BETA = %.4f' % BETA)
# logger.info('LAMBDA1 = %.4f' % LAMBDA1)
# logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
# logger.info('MU = %.4f' % MU)
# logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
# logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')
