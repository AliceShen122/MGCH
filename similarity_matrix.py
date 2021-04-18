#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import kk


def similarity_matrix():
    query_label = np.expand_dims(kk.test_label_set.astype(dtype=np.int), 1)  # (2000, 1, 80)
    retrieval_label = np.expand_dims(kk.train_label_set.astype(dtype=np.int), 0)  # (1, 50000, 80)
    similarity_matrix = np.bitwise_and(query_label, retrieval_label)  # (2000,50000,80)
    similarity_matrix = np.sum(similarity_matrix, axis=2)  # (2000,50000)
    similarity_matrix[similarity_matrix >= 1] = 1  # (2000,50000)
    return similarity_matrix
