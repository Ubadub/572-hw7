#!/nopt/python-3.6/bin/python3.6

# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch_maxent.py

import itertools
import sys

import numpy as np

from beamsearch import beam_search_write
from maxent import MaxEntModel

CLS_LBL_SPLIT = '-'
FPRINT = lambda f: lambda *x, **y: print(*x, file=f, **y)
SEP = '%'*5

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

        yield raw_lbls, word_gold_classes, sentM

def read_boundaries(boundary_stream):
    return [int(l) for l in boundary_stream if l]

# cat $test_data | ./beamsearch_maxent.py $boundary_file $model_file $sys_output $beam_size $topN $topK
def main():
    boundary_f, model_f, sys_output, beam_size, topN, topK = sys.argv[1:7]
    beam_size = int(beam_size)
    topN = int(topN)
    topK = int(topK)
    with open(model_f, 'r') as f:
        me_model = MaxEntModel.from_model_file_stream(f)

    with open(boundary_f, 'r') as f:
        boundaries = read_boundaries(f)

    mapping = me_model.cls_idx2lbl, me_model.cls_lbl2idx, me_model.feat_idx2lbl, me_model.feat_lbl2idx
    
    with open(sys_output, 'w') as f_out:
        fprint = FPRINT(f_out)
        fprint(SEP, "test data:")
        i = 1
        for raw_lbls, gold_y, X in  process_input(boundaries, mapping):
            beam_search_write(fprint, raw_lbls, gold_y, X, me_model, beam_size, topN, topK)
            if i % 3 == 0:
                f_out.flush()
            i += 1

if __name__ == '__main__':
    main()
