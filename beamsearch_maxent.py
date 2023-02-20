#!/nopt/python-3.6/bin/python3.6

# Abhinav Patil
# Feb 22, 2023
# LING 572 - Winter 23 - Fei Xia
# HW 7 Question 1
# beamsearch_maxent.py

from maxent import MaxEntModel

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
