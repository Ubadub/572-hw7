# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch.py

import itertools
import re

import numpy as np

from maxent import MaxEntModel

BOS_TAG = "BOS"
# DOUBLE_BOS_TAG = "BOS+BOS"
PREVW_FEAT_PREFIX = "prevW="
PREV2W_FEAT_PREFIX = "prev2W="
PREVT_FEAT_PREFIX = "prevT="
PREV2T_FEAT_PREFIX = "prevTwoTags="

# INIT_PREV_T = f"{PREVT_FEAT_PREFIX}{BOS_TAG}"
# INIT_PREV2TAGS = f"{PREV2T_FEAT_PREFIX}{DOUBLE_BOS_TAG}"

# INIT_PREV_T = BOS_TAG
# INIT_PREV2TAGS = DOUBLE_BOS_TAG

class Node:

    def __init__(self, tag_lbl, prob, prevNode=None):
        super(Node, self).__init__()
        # self._tag_idx = tag_idx
        self._tag_lbl = tag_lbl
        self._prob = prob
        self._prevNode = prevNode

        if prevNode is None:
            self._cum_prob = self._prob
        else:
            self._cum_prob = prevNode._cum_prob * self._prob
        self._lg_prob = np.log10(self._prob)
        self._lg_cum_prob = np.log10(self._cum_prob)

    @property
    def tag_lbl(self):
        return self._tag_lbl

    @property
    def parent(self):
        return self._prevNode

    @property
    def prob(self):
        return self._prob

    # @property
    # def lg_prob(self):
    #     return self._lg_prob

    @property
    def lg_cum_prob(self):
        return self._lg_cum_prob

    @property
    def cum_prob(self):
        return self._cum_prob

    def __str__(self):
        return f"[[ {self.tag_lbl}:\tProb={self._prob}\tParent={self.parent.tag_lbl} ]]"

def build_word_vec(x_vec_orig, prevT_feat, prevTwoTags_feat, feat_lbl2idx):
    # if prevT_feat in feat_lbl2idx:
    # if prevTwoTags_feat in feat_lbl2idx:
    prevT_feat_idx = feat_lbl2idx.get(prevT_feat, None)
    prevTwoTags_feat_idx = feat_lbl2idx.get(prevTwoTags_feat, None)

    # print(prevT_feat, ":", prevT_feat_idx)
    # print(prevTwoTags_feat,":", prevTwoTags_feat_idx)

    x_vec = x_vec_orig.copy()

    if prevT_feat_idx is not None:
        x_vec[prevT_feat_idx] = True
    #     print("Added feature for", prevT_feat)
    # else:
    #     print("Not adding feature for", prevT_feat)

    if prevTwoTags_feat_idx is not None:
        x_vec[prevTwoTags_feat_idx] = True
    #     print("Added feature for", prevTwoTags_feat)
    # else:
    #     print("Not adding feature for", prevTwoTags_feat)

    return x_vec

def beam_search(model, sentX, beam_size, topN, topK):
    """
    Args:
        sentX:
            shape: (number of words in sentence, number of features in model)
    """
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = model.cls_idx2lbl, model.cls_lbl2idx, model.feat_idx2lbl, model.feat_lbl2idx
    # print(cls_lbl2idx)
    # print(feat_lbl2idx)
    res = np.zeros((sentX.shape[0],), dtype=np.uint64) # POS tags as class indices
    probs = np.empty((sentX.shape[0],))

    # prevT_lbl = INIT_PREV_T
    # prevTwoTags_lbl = INIT_PREV2TAGS
    # prevT = BOS_TAG
    # prevTwoTags = DOUBLE_BOS_TAG

    root_node = Node(BOS_TAG, 1.0, prevNode=Node(BOS_TAG, 1.0))
    # nodes = [set() for i in itertools.repeat(None, len(sentX))]
    nodes = [set() for i in range(len(sentX))]
    # paths = {} # nodes from i-1

    # nodes[0].append(Node(cls_lbl2idx[BOS_TAG]))
    for word_idx, wordX in enumerate(sentX):
        # print(f"========== Word Index: {word_idx} ==========")
        if word_idx == 0:
            prev_level_nodes = [root_node]
        else:
            prev_level_nodes = nodes[word_idx - 1]

        for parent_node in prev_level_nodes:
            # print(f"Expanding children of {parent_node} at level {word_idx}...")
            parentTag = parent_node.tag_lbl
            grandparentTag = parent_node.parent.tag_lbl
            prevT_feat = f"{PREVT_FEAT_PREFIX}{parentTag}"
            prevTwoTags_feat = f"{PREV2T_FEAT_PREFIX}{grandparentTag}+{parentTag}"

            # print(f"Parent tag: {parentTag}. Grandparent tag: {grandparentTag}. Reconstructing features: {prevT_feat} and {prevTwoTags_feat}")

            x_vec = build_word_vec(wordX, prevT_feat, prevTwoTags_feat, feat_lbl2idx)

            # new_nodes = set()
            tag_lbl_arr = np.array(cls_idx2lbl)
            # tag_idxs = range(model.num_classes)

            tag_probs = model.calc_prob_feat_given_cls(x_vec)
            tag_probs_sort_idxs = np.argsort(tag_probs)
            # print(f"For word at index {word_idx}:")
            # print(*zip(tag_lbl_arr, tag_probs), sep='\n')
            
            # tag_lbls_sorted = tag_lbl_arr[tag_probs_sort_idxs]
            # tag_probs_sorted = tag_probs[tag_probs_sort_idxs]
            # print("Sorted:")
            # print(*zip(tag_lbls_sorted, tag_probs_sorted), sep='\n')
            
            # topN_tag_lbls = tag_lbls_sorted[-topN:]
            # topN_tag_probs = tag_probs_sorted[-topN:]
            # print(f"Top {topN}:")
            # print(*zip(topN_tag_lbls, topN_tag_probs), sep='\n')
            # print("###########")

            topN_tag_idxs = tag_probs_sort_idxs[-topN:]

            for tag_idx in topN_tag_idxs:
                nodes[word_idx].add(Node(cls_idx2lbl[tag_idx], tag_probs[tag_idx], prevNode=parent_node))

        if word_idx > 0:
            # print(f"Pruning. Before: {len(nodes[word_idx])} elements.")
            nodes[word_idx] = prune_nodes(nodes[word_idx], topK, beam_size)
            # print(f"After: {len(nodes[word_idx])} elements.")

    best_terminal_node = sorted(nodes[-1], key=lambda x: x.cum_prob)[-1]
    curr_node = best_terminal_node
    for i in range(sentX.shape[0]-1, -1, -1):
        res[i] = cls_lbl2idx[curr_node.tag_lbl]
        probs[i] = curr_node.prob
        curr_node = curr_node.parent

    # path = []
    # while curr_node and curr_node.parent is not None and curr_node.parent is not None:
    #     path.append(curr_node)
    #     curr_node = curr_node.parent

    # print(*(str(n) for s in nodes for n in s), sep='\n')
    # print(*(str(n) for n in path), sep='\n')
    return res, probs

def prune_nodes(nodes, topK, beam_size):
    keep = set()
    sorted_nodes = sorted(nodes, key=lambda x: x.cum_prob, reverse=True)
    best_lg_cum_prob = sorted_nodes[0].lg_cum_prob
    keep = {n for n in sorted_nodes[:topK] if n.lg_cum_prob + beam_size >= best_lg_cum_prob}
    return keep

def output_classification_result(printer, model, raw_inst_lbls, gold_tags, tags, probs, prec=6):
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = model.cls_idx2lbl, model.cls_lbl2idx, model.feat_idx2lbl, model.feat_lbl2idx

    for inst_name, gold_cls_idx, sys_cls_idx, prob in zip(raw_inst_lbls, gold_tags, tags, probs):
        gold_cls_lbl = cls_idx2lbl[gold_cls_idx]
        sys_cls_lbl = cls_idx2lbl[sys_cls_idx]
        printer(inst_name, gold_cls_lbl, sys_cls_lbl, prob)

# cat $test_data | ./beamsearch_maxent.py $boundary_file $model_file $sys_output $beam_size $topN $topK
def beam_search_write(f_out, raw_lbls, gold_y, X, me_model, beam_size, topN, topK):
    y_hat, probs = beam_search(me_model, X, beam_size, topN, topK)
    output_classification_result(f_out, me_model, raw_lbls, gold_y, y_hat, probs)



