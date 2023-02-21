# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# node.py

import numpy as np

class Node:
    def __init__(self, tag_lbl, prob, prevNode=None):
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

    @property
    def lg_cum_prob(self):
        return self._lg_cum_prob

    @property
    def cum_prob(self):
        return self._cum_prob

    def __str__(self):
        return f"[[ {self.tag_lbl}:\tProb={self._prob}\tParent={self.parent.tag_lbl} ]]"