#!/nopt/python-3.6/bin/python3.6

# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch.py

from collections import defaultdict
import itertools
import re
import sys

import numpy as np

from maxent import MaxEntModel

MOD_FILE_CLASS_HEADER_PAT = re.compile(r'^\s*FEATURES\s+FOR\s+CLASS\s+(\S+)\s*$')
CLS_LBL_SPLIT = '-'
DEFAULT_FEAT_LBL = '<default>'
FEAT_SPLIT = ':'
INIT_PREV_T = "BOS"
INIT_PREV2TAGS = "BOS+BOS1"
SEP = '%'*5
FPRINT = lambda f: lambda *x, **y: print(*x, file=f, **y)

ONE_VEC = np.array([True])

def process_input(boundaries, mapping, input_stream=sys.stdin):
    inst_cls_idxs = []
    X = []
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = mapping
    num_feats = len(feat_lbl2idx)

    lines = (l for l in (line.strip() for line in input_stream) if l)

    for boundary in boundaries:
        inst_lines = itertools.islice(lines, boundary)
        raw_lbls = []
        word_gold_classes = []
        word_feat_vecs = []
        for line in inst_lines:
            inst = line.split()
            inst_name = inst[0]
            raw_lbls.append(inst_name)
            inst_word = inst_name.split(CLS_LBL_SPLIT)[-1]

            gold_lbl = inst[1]
            gold_cls_idx = cls_lbl2idx[gold_lbl]
            word_gold_classes.append(gold_cls_idx)

            feats = inst[2::2]
            wts = inst[3::2]
            feat_idxs = [feat_lbl2idx[f] for f,wt in zip(feats, wts) if int(wt) > 0]

            word_feat_vec = np.zeros((num_feats,), dtype=bool)
            word_feat_vec[0] = True # default feature
            word_feat_vec[feat_idxs] = True
            word_feat_vecs.append(word_feat_vec)

        sentM = np.row_stack(word_feat_vecs)
#        print(sentM)
#        print(sentM.shape)
#        print(gold_classes)
#        print(gold_classes.shape)
        yield raw_lbls, word_gold_classes, sentM

def read_boundaries(boundary_stream):
    return [int(l) for l in boundary_stream if l]

def output_classification_result(output_stream, header, raw_inst_lbls, gold_tags, tags, probs, mapping, prec=6):
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = mapping
    fprint = FPRINT(output_stream)
    fprint(SEP, header)
    for inst_name, gold_cls_idx, sys_cls_idx, prob in zip(raw_inst_lbls, gold_tags, tags, probs):
        gold_cls_lbl = cls_idx2lbl[gold_cls_idx]
        print(inst_name)
        print(gold_cls_idx)
        print(sys_cls_idx)
        sys_cls_lbl = cls_idx2lbl[sys_cls_idx]
        fprint(inst_name, gold_cls_lbl, sys_cls_lbl, prob)
#        idx_sort = inst_probs.argsort()[::-1]
#        cls_lbls_cond_probs = np.dstack((cls_idx2lbl, inst_probs))[:,idx_sort].flatten()
#        fprint(f'array:{inst_idx}', gold_tag, *cls_lbls_cond_probs)

def beam_search(model, sentX, beam_size, topN, topK):
    """
    Args:
        sentX:
            shape: (number of words in sentence, number of features in model)
    """
    cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = model.cls_idx2lbl, model.cls_lbl2idx, model.feat_idx2lbl, model.feat_lbl2idx
    res = np.zeros((sentX.shape[0],), dtype=np.uint64) # POS tags as class indices
    probs = np.empty((sentX.shape[0],))
    prevT = cls_lbl2idx[INIT_PREV_T]
    prevTwoTags = cls_lbl2idx[INIT_PREV2TAGS]
    paths = {} # nodes from i-1
    first_word_topN_tags
    for wordX in sentX:
        x_vec = wordX.copy()
        wordX[prevT] = True
        wordX[prevTwoTags] = True
        if prevTwoTags == cls_lbl2idx[INIT_PREV2TAGS]:
            pass
        for tag in range(len(cls_idx2lbl)):
            if tag == 1:
                pass
    return res, probs

def main():
    boundary_f, model_f, sys_output, beam_size, topN, topK = sys.argv[1:7]
    with open(model_f, 'r') as f:
        me_model = MaxEntModel.from_model_file_stream(f)

    with open(boundary_f, 'r') as f:
        boundaries = read_boundaries(f)

    mapping = me_model.cls_idx2lbl, me_model.cls_lbl2idx, me_model.feat_idx2lbl, me_model.feat_lbl2idx
    
    with open(sys_output, 'w') as f_out:
        for raw_lbls, gold_y, X in  process_input(boundaries, mapping):
            y_hat, probs = beam_search(me_model, X, beam_size, topN, topK)
            output_classification_result(f_out, "test data:", raw_lbls, gold_y, y_hat, probs, mapping)
        
#        fprint(

#    cls_lbl2idx = me_model.cls_lbl2idx 
#    feat_lbl2idx = me_model.feat_lbl2idx 
#
#    X_test, y_test = process_input(sys.stdin, mapping)
#
#    test_probs = me_model.calc_probs(X_test)
#    with open(sys_output, 'w') as sys_file:
#        output_classification_result(sys_file, 'test data:', test_probs, y_test, mapping)
#
#    test_confusionM = me_model.build_confusion_mat(test_probs, y_test)
#    me_model.output_confusion_mat(test_confusionM, 'test')

if __name__ == '__main__':
    main()
