import numpy as np
import torch
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve

def find_best_thresholds(y_true, y_score):
    thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, th = precision_recall_curve(y_true[:, i], y_score[:, i])
        f1s = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = th[np.argmax(f1s)] if len(th) > 0 else 0.5
        thresholds.append(best_thresh)
    return np.array(thresholds)

def eval_multilabel_classification(
    model, train_all, train_labels, test_all, test_labels, val_all, val_labels,
    n_channels, encoding_protocol='full_series', eval_protocol='linear'
):
    if encoding_protocol == 'full_series':
        assert train_labels.ndim == 2  # Must be multi-label
        # print("aaa")
        train_repr = model.encode(train_all, n_channels, encoding_window='full_series')
        test_repr = model.encode(test_all, n_channels, encoding_window='full_series')
        val_repr = model.encode(val_all, n_channels, encoding_window='full_series')
        print(train_repr.shape)
        print(train_labels.shape)

    elif encoding_protocol == 'timedim':
        assert train_all.shape[1] == test_all.shape[1]
        T = train_all.shape[1]
        k = 10
        w = (T // k) if T > k else 1
        train_repr = model.encode(train_all, n_channels, encoding_window=w)
        test_repr = model.encode(test_all, n_channels, encoding_window=w)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        test_repr = test_repr.reshape(test_repr.shape[0], -1)

    # Select the classifier fitting function
    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn_multi_label
    else:
        raise ValueError(f'Unknown evaluation protocol: {eval_protocol}')

    if train_repr.ndim == 3:
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
    if test_repr.ndim == 3:
        test_repr = test_repr.reshape(test_repr.shape[0], -1)
    # Train classifier on representations
    clf = fit_clf(train_repr, train_labels)
    print("aaaa")
    

    #Tune the classification threshold
    # try:
    #     val_scores = clf.predict_proba(val_repr)
    # except AttributeError:
    #     val_scores = clf.decision_function(val_repr)


    # thresholds = find_best_thresholds(val_labels, val_scores)
    # # y_pred = clf.predict(test_repr)
    try:
        y_score = clf.predict_proba(test_repr)
    except:
        y_score = clf.decision_function(test_repr)

    # y_pred = (y_score >= thresholds).astype(int)
    # print(thresholds)

    threshold = 0.3
    y_pred = (y_score >= threshold).astype(int)



    # Evaluate
    f1_micro = f1_score(test_labels, y_pred, average='micro')
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    precision_micro = precision_score(test_labels, y_pred, average='micro')
    recall_micro = recall_score(test_labels, y_pred, average='micro')
    accuracy = accuracy_score(test_labels, y_pred)
    
    auprc = average_precision_score(test_labels, y_score, average='macro')
    
    # AUROC per class
    auroc_per_class = []
    for i in range(test_labels.shape[1]):
        try:
            score = roc_auc_score(test_labels[:, i], y_score[:, i])
        except ValueError:
            score = float('nan')
        auroc_per_class.append(score)
    auroc_macro = np.nanmean(auroc_per_class)
    
    auroc_macro = np.nanmean(auroc_per_class)

    return y_score, {
    'accuracy': accuracy,
    'precision_micro': precision_micro,
    'recall_micro': recall_micro,
    'f1_micro': f1_micro,
    'f1_macro': f1_macro,
    'auprc': auprc,
    'auroc_macro': auroc_macro,
    'auroc_per_class': auroc_per_class,
}




def eval_classification(model, train_all, train_labels, test_all, test_labels, n_channels,encoding_protocol='full_series', eval_protocol='linear'):

    if encoding_protocol == 'full_series':
        # 'full_series' encodes time series in 1 vector (no temporal dimension). This is the default and simplest setting.
        # It should be sufficient for most applications, but for maximum performance use 'timedim'.
        assert train_labels.ndim == 1 or train_labels.ndim == 2
        train_repr = model.encode(train_all, n_channels, encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_all, n_channels, encoding_window='full_series' if train_labels.ndim == 1 else None)
        # print(train_repr.shape)
        # print(test_repr.shape)
        
    elif encoding_protocol == 'timedim':
        # 'timedim' encodes time series at a user-specified temporal granularity, resulting in higher-dimensional representations
        # to classify, but more easily separable. This can boost performance, but it more computationally expensive and requires
        # tuning hyper-parameter k.
        assert train_all.shape[1] == test_all.shape[1]
        T = train_all.shape[1]
        k = 10
        w = (T // k) if T > k else 1
        train_repr = model.encode(train_all, n_channels, encoding_window=w if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_all, n_channels, encoding_window=w if train_labels.ndim == 1 else None)

        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        test_repr = test_repr.reshape(test_repr.shape[0], -1)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    # print(train_repr.shape)
    # print(train_labels.shape)
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    # print(train_repr.shape)
    # print(train_labels.shape)
    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }
