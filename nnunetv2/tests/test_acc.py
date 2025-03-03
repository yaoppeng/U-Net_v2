from sklearn.metrics import balanced_accuracy_score

y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

balanced_accuracy_score(y_true, y_pred)

"""
[
    true negatives, false negatives
    true positives, false positives
]

1/2(TN/(TN+FN) + FP/(TP+FP)
"""