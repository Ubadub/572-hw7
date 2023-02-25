#!/bin/sh

# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch_maxent.sh

if [ $# -ne 7 ]; then
    echo "Usage: beamsearch_maxent.sh <test_data> <boundary_file> <model_file> <sys_output> <beam_size> <topN> <topK>"
else
    test_data=$1
    boundary_file=$2
    model_file=$3
    sys_output=$4
    beam_size=$5
    topN=$6
    topK=$7
    cat $test_data | ./beamsearch_maxent.py $boundary_file $model_file $sys_output $beam_size $topN $topK
fi
