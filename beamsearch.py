# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch.py

import itertools
import re

import numpy as np

from maxent import MaxEntModel
from node import Node

BOS_TAG = "BOS"
PREVT_FEAT_PREFIX = "prevT="
PREV2T_FEAT_PREFIX = "prevTwoTags="

def build_word_vec(x_vec_orig, prevT_feat, prevTwoTags_feat, feat_lbl2idx):
    prevT_feat_idx = feat_lbl2idx.get(prevT_feat, None)
    prevTwoTags_feat_idx = feat_lbl2idx.get(prevTwoTags_feat, None)

    x_vec = x_vec_orig.copy()

    if prevT_feat_idx is not None:
        x_vec[prevT_feat_idx] = True

    if prevTwoTags_feat_idx is not None:
        x_vec[prevTwoTags_feat_idx] = True

    return x_vec

def prune_nodes(nodes, topK, beam_size):
    keep = set()
    sorted_nodes = sorted(nodes, key=lambda x: x.cum_prob, reverse=True)
    best_lg_cum_prob = sorted_nodes[0].lg_cum_prob
    keep = {n for n in sorted_nodes[:topK] if n.lg_cum_prob + beam_size >= best_lg_cum_prob}
    return keep

def beam_search(model, sentX, beam_size, topN, topK):
    """
    Args:
        sentX:
            shape: (number of words in sentence, number of features in model)
    """
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = model.cls_idx2lbl, model.cls_lbl2idx, model.feat_idx2lbl, model.feat_lbl2idx
    res = np.zeros((sentX.shape[0],), dtype=np.uint64) # POS tags as class indices
    probs = np.empty((sentX.shape[0],))

    root_node = Node(BOS_TAG, 1.0, prevNode=Node(BOS_TAG, 1.0))
    nodes = [set() for i in range(len(sentX))]

    for word_idx, wordX in enumerate(sentX):
        if word_idx == 0:
            prev_level_nodes = [root_node]
        else:
            prev_level_nodes = nodes[word_idx - 1]

        for parent_node in prev_level_nodes:
            parentTag = parent_node.tag_lbl
            grandparentTag = parent_node.parent.tag_lbl
            prevT_feat = f"{PREVT_FEAT_PREFIX}{parentTag}"
            prevTwoTags_feat = f"{PREV2T_FEAT_PREFIX}{grandparentTag}+{parentTag}"

            x_vec = build_word_vec(wordX, prevT_feat, prevTwoTags_feat, feat_lbl2idx)

            tag_lbl_arr = np.array(cls_idx2lbl)

            tag_probs = model.calc_prob_feat_given_cls(x_vec)
            tag_probs_sort_idxs = np.argsort(tag_probs)

            topN_tag_idxs = tag_probs_sort_idxs[-topN:]

            for tag_idx in topN_tag_idxs:
                nodes[word_idx].add(Node(cls_idx2lbl[tag_idx], tag_probs[tag_idx], prevNode=parent_node))

        if word_idx > 0:
            nodes[word_idx] = prune_nodes(nodes[word_idx], topK, beam_size)

    best_terminal_node = sorted(nodes[-1], key=lambda x: x.cum_prob)[-1]
    curr_node = best_terminal_node
    for i in range(sentX.shape[0]-1, -1, -1):
        res[i] = cls_lbl2idx[curr_node.tag_lbl]
        probs[i] = curr_node.prob
        curr_node = curr_node.parent

    return res, probs

def output_classification_result(printer, model, raw_inst_lbls, gold_tags, tags, probs, prec=6):
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = model.cls_idx2lbl, model.cls_lbl2idx, model.feat_idx2lbl, model.feat_lbl2idx
    num_right = 0
    for inst_name, gold_cls_idx, sys_cls_idx, prob in zip(raw_inst_lbls, gold_tags, tags, probs):
        gold_cls_lbl = cls_idx2lbl[gold_cls_idx]
        sys_cls_lbl = cls_idx2lbl[sys_cls_idx]
        printer(inst_name, gold_cls_lbl, sys_cls_lbl, prob)
        if gold_cls_lbl == sys_cls_lbl:
            num_right += 1

    return num_right

def beam_search_write(f_out, raw_lbls, gold_y, X, me_model, beam_size, topN, topK):
    y_hat, probs = beam_search(me_model, X, beam_size, topN, topK)
    return output_classification_result(f_out, me_model, raw_lbls, gold_y, y_hat, probs)