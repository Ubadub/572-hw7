# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch_maxent.sh

import re
import sys

from collections import defaultdict

import numpy as np

DEFAULT_FEAT_LBL = '<default>'
FEAT_SPLIT = ':'
FPRINT = lambda f: lambda *x, **y: print(*x, file=f, **y)
MOD_FILE_CLASS_HEADER_PAT = re.compile(r'^\s*FEATURES\s+FOR\s+CLASS\s+(\S+)\s*$')
ONE_VEC = np.array([True])
SEP = '%'*5

def input_data_matrix(input_stream=sys.stdin, featt_split=':', mapping=None, dtype=np.int64):
    inst_cls_lbls = []
    inst_feat_lbls = []
    if not mapping:
        cls_lbls = set()
        feat_lbls = { DEFAULT_FEAT_LBL }

    for line in input_stream:
        if line:
            inst = line.split()
            cls_lbl = inst[0]
            feat_cnts = inst[1:]
            inst_cls_lbls.append(cls_lbl)
            inst_feat_lbls.append(defaultdict(int))

            for feat_cnt in feat_cnts:
                feat_lbl, feat_cnt = feat_cnt.split(FEAT_SPLIT)
                feat_cnt = int(feat_cnt)
                if not mapping or feat_lbl in mapping[-1]:
                    inst_feat_lbls[-1][feat_lbl] += feat_cnt
            if not mapping:
                cls_lbls.add(cls_lbl)
                feat_lbls.update(inst_feat_lbls[-1].keys())
    if mapping:
        cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = mapping
    else:
        cls_idx2lbl = sorted(list(cls_lbls))
        cls_lbl2idx = { lbl : idx for idx,lbl in enumerate(cls_idx2lbl) }

        feat_idx2lbl = sorted(list(feat_lbls))
        feat_lbl2idx = { lbl : idx for idx,lbl in enumerate(feat_idx2lbl) }

        mapping = cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx

    y = np.c_[[cls_lbl2idx[cls_lbl] for cls_lbl in inst_cls_lbls]]
    X = np.zeros((y.shape[0], len(feat_lbl2idx)), dtype=dtype)
    for inst_idx, inst_feat_cnts in enumerate(inst_feat_lbls):
        for feat_lbl, feat_cnt in inst_feat_cnts.items():
            feat_idx = feat_lbl2idx[feat_lbl]
            X[inst_idx, feat_idx] += feat_cnt

    X[:,0] = 1 # default feature

    return X, y, mapping

def lg_prob_normalize(lg_probs):
    maxes = lg_probs.max(axis=-1, keepdims=True) # so it works for 1 dim arrays too
    norm_const = maxes + np.log10((10.0 ** (lg_probs - maxes)).sum(axis=-1, keepdims=True))
    return lg_probs - norm_const

class MaxEntModel:
    @classmethod
    def from_model_file_stream(cls, input_stream=sys.stdin):
        def _id_counter(counter=0):
            def _():
                    nonlocal counter
                    new_state_id = counter
                    counter += 1
                    return new_state_id
            return _

        cls_lbl2idx = defaultdict(_id_counter())
        feat_lbl2idx = defaultdict(_id_counter())
        feat_lbl2idx[DEFAULT_FEAT_LBL] # create 0th index, associate with default feature
        cls_feat_wts = defaultdict(dict) # cls_feat_wts[cls_idx][feat_idx] = lambda_{cls,feat}

        curr_cls_idx = -1
        for line in input_stream:
            line = line.strip()
            if line:
                cls_headers = MOD_FILE_CLASS_HEADER_PAT.findall(line)
                if len(cls_headers) == 1:
                    curr_cls_idx = cls_lbl2idx[cls_headers[0]]
                elif curr_cls_idx >=0 and not cls_headers:
                    feat_lbl, wt = line.split()
                    wt = float(wt)
                    feat_idx = feat_lbl2idx[feat_lbl]
                    cls_feat_wts[curr_cls_idx][feat_idx] = wt
                else:
                    print('Invalid format for model!', file=sys.stderr)
                    return

        cls_lbl2idx = dict(cls_lbl2idx)
        cls_cnt = len(cls_lbl2idx)
        cls_idx2lbl = [t[0] for t in sorted(cls_lbl2idx.items(), key=lambda x:x[1])]

        feat_lbl2idx = dict(feat_lbl2idx)
        feat_cnt = len(feat_lbl2idx)
        feat_idx2lbl = [t[0] for t in sorted(feat_lbl2idx.items(), key=lambda x:x[1])]
        cls_feat_wtM = np.zeros((cls_cnt, feat_cnt))
        for cls_idx in cls_feat_wts:
            for feat_idx, wt in cls_feat_wts[cls_idx].items():
                cls_feat_wtM[cls_idx][feat_idx] = wt

        return MaxEntModel(cls_feat_wtM, cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx)

    def __init__(self, cls_feat_wtM, cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx):
        self._cls_feat_wtM = cls_feat_wtM
        self._cls_idx2lbl = cls_idx2lbl
        self._cls_lbl2idx = cls_lbl2idx
        self._feat_idx2lbl = feat_idx2lbl
        self._feat_lbl2idx = feat_lbl2idx

    @property
    def num_classes(self):
        return len(self.cls_idx2lbl)

    @property
    def num_features(self):
        return len(self.feat_idx2lbl)

    @property
    def cls_idx2lbl(self):
        return self._cls_idx2lbl

    @property
    def cls_lbl2idx(self):
        return self._cls_lbl2idx

    @property
    def feat_idx2lbl(self):
        return self._feat_idx2lbl

    @property
    def feat_lbl2idx(self):
        return self._feat_lbl2idx

    # def add_class(self, cls_lbl):
    #     if cls_lbl not in self.cls_lbl2idx:
    #         cls_idx = self.num_classes
    #         self._cls_idx2lbl.append(cls_lbl)
    #         self._cls_lbl2idx[cls_lbl] = cls_idx
    #     return self.cls_lbl2idx[feat_lbl]

    # def add_feature(self, feat_lbl):
    #     if feat_lbl not in self.feat_lbl2idx:
    #         feat_idx = self.num_features
    #         self._feat_idx2lbl.append(feat_lbl)
    #         self._feat_lbl2idx[feat_lbl] = feat_idx
    #     return self.feat_lbl2idx[feat_lbl]

    def calc_prob_feat_given_cls(self, X):
        weighted_feats = X * self._cls_feat_wtM
        numer = np.e ** lg_prob_normalize(weighted_feats.sum(axis=1))
        norm_term = numer.sum()
        return numer/norm_term

    def calc_probs(self, testX):
        """
        Given N x F matrix testX,
        return a N x Y matrix resM where:
            resM[n,y] = P(y | testX[n])
            (that is, the conditional probability for each class given each instance)
        N = number of training instances
        Y = number of classes
        n = a training instance (0 <= n < N)
        y = a class (0 <= y < Y)
        """
        resM = np.zeros((len(testX), self.num_classes))
        for inst_idx, inst_feats in enumerate(testX):
            resM[inst_idx] = self.calc_prob_feat_given_cls(inst_feats)

        return resM

    # def build_confusion_mat(self, inst_class_probs, y_true):
    #     confusionM = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    #     y_hat_idxs = inst_class_probs.argmax(axis=1)
    #     for inst_idx, cls_probs in enumerate(inst_class_probs):
    #         true_cls_idx = y_true[inst_idx][0]
    #         true_cls_lbl = self.cls_idx2lbl[true_cls_idx]
    #         y_hat_cls_idx = cls_probs.argmax()
    #         confusionM[true_cls_idx, y_hat_cls_idx] += 1
    #     return confusionM

    # def output_confusion_mat(self, confusionM, header, footer='',  tbl_header_spaces=12, output_stream=sys.stdout):
    #     num_correct = confusionM.diagonal().sum()
    #     tot = confusionM.sum()
    #     acc = num_correct / tot
    #     fprint = FPRINT(output_stream)
    #     fprint(f'Confusion matrix for the {header.lower()} data:')
    #     fprint('row is the truth, column is the system output')
    #     fprint()
    #     fprint(' ' * tbl_header_spaces, *self.cls_idx2lbl)
    #     for idx, r in enumerate(confusionM):
    #         fprint(self.cls_idx2lbl[idx], *confusionM[idx,:])
    #     fprint()
    #     fprint(f' {header.capitalize()} accuracy={acc}')
    #     fprint(footer, end='')

    # def output_classification_result(output_stream, header, probs_per_class, true_lbls, mapping, prec=6):
    #     cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx = mapping
    #     fprint = FPRINT(output_stream)
    #     fprint(SEP, header)
    #     for inst_idx in range(len(probs_per_class)):
    #         true_cls_lbl = cls_idx2lbl[true_lbls[inst_idx][0]]
    #         inst_probs = probs_per_class[inst_idx]
    #         idx_sort = inst_probs.argsort()[::-1]
    #         str_inst_probs = np.array([f'{s:0.{prec}f}' for s in inst_probs])
    #         cls_lbls_cond_probs = np.dstack((cls_idx2lbl, str_inst_probs))[:,idx_sort].flatten()
    #         fprint(f'array:{inst_idx}', true_cls_lbl, *cls_lbls_cond_probs)

    # def write(self, model_file):
    #     with open(model_file, 'w') as f:
    #         fprint = FPRINT(f)
    #         fprint(SEP, 'prior prob P(c)', SEP)
    #         for cls_idx, cls_lbl in enumerate(self.cls_idx2lbl):
    #             fprint(cls_lbl, self.cls_probM[cls_idx,0], self.lg_cls_probM[cls_idx,0])

    #         fprint(SEP, 'conditional prob P(f|c)', SEP)
    #         for cls_idx, cls_lbl in enumerate(self.cls_idx2lbl):
    #             fprint(SEP, 'conditional prob P(f|c)', f'c={cls_lbl}', SEP)
    #             for feat_idx, feat_lbl in enumerate(self.feat_idx2lbl):
    #                 fprint(feat_lbl, cls_lbl, self.cond_probM[cls_idx,feat_idx], self.lg_cond_probM[cls_idx,feat_idx])


class UniformMaxEntModel(MaxEntModel):
    def __init__(self, cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx):
        super().__init__(None, cls_idx2lbl, cls_lbl2idx, feat_idx2lbl, feat_lbl2idx)
        self._cls_feat_wtM = np.zeros((self.num_classes, self.num_features)) # no need for log
        # assert DEFAULT_FEAT_LBL in self.feat_lbl2idx, f"DEFAULT_FEAT_LBL {DEFAULT_FEAT_LBL} not in feat labels"
        # assert self.feat_lbl2idx[DEFAULT_FEAT_LBL] == 0, f"DEFAULT_FEAT_LBL {DEFAULT_FEAT_LBL} indexed to {self.feat_lbl2idx[DEFAULT_FEAT_LBL]}, not 0"