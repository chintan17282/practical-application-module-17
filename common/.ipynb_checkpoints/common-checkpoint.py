from sklearn.metrics import accuracy_score, recall_score, precision_score

def precision_thresh(predict_probs, y_test, thresh):
    preds = np.where(predict_probs >= thresh, 'yes', 'no')    
    return precision_score(y_test, preds, pos_label='yes')
