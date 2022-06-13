#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Paul Houssel
# License: MIT License

"""
This python script serves to evaluation the models trained in the experiment D-bis.
The evalution metrics are computed for the models of every user, to obtain the average Equal Error Rate (EER),
the Balanced Accuracy, and the weighted AUC score.
The training.py script should be executed before this script.
"""

from os import path, listdir, environ
from tensorflow import keras
from numpy import absolute, nanargmin, array, load
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from statistics import mean

# Ignore the Tensorflow Informations and Warnings
environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def find_EER(far, frr):
   far_optimum = 0
   frr_optimum = 0
   x = absolute((array(far) - array(frr)))
   y = nanargmin(x)
   far_optimum = far[y]
   frr_optimum = frr[y]
   return far_optimum, frr_optimum

def eval_binary_classifier(y_true, y_pred, class_weights=None):
    prob_labels = array(y_pred).ravel()
    pred_labels = prob_labels > 0.5

    true_labels = y_true.astype(int)
    pred_labels = pred_labels.astype(int)

    # Compute the EER metric
    fpr, tpr,  threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr # get FNR , however FPR is same as FAR
    far_optimum, frr_optimum = find_EER(fpr, fnr)
    EER = max(far_optimum, frr_optimum)

    return {
        'balanced_acc': balanced_accuracy_score(true_labels, pred_labels),
        'auc_weighted': roc_auc_score(y_true, y_pred, average='weighted'),
        'eer':EER
    }

# Directory of the saved models
base_dir = "models"

models_auc = {"7": [], "9":[], "12":[], "15":[],"16":[],"20":[], "21":[], "23":[],'29':[],"35":[]}
models_acc = {"7": [], "9":[], "12":[], "15":[],"16":[], "20":[], "21":[], "23":[],'29':[],"35":[]}
models_eer = {"7": [], "9":[], "12":[], "15":[],"16":[], "20":[], "21":[], "23":[],'29':[],"35":[]}
models_auc_mean = {}
models_acc_mean = {}
models_eer_mean = {}

# For every saved model, we compute the three metrics, and save them in the given dictionarries
for model_file_name in listdir(base_dir):
    loaded_model = keras.models.load_model(path.join(base_dir, model_file_name))
    # obtain the testing set from the given model
    X_test = load(path.join("models-testingsets", model_file_name[:-3] + "-test-X.npy"))
    Y_test = load(path.join("models-testingsets", model_file_name[:-3] + "-test-Y.npy"))
    # Extract the user id
    user = model_file_name.split("-")[0].split("r")[1]
    Y_pred = loaded_model.predict(X_test).ravel()
    metrics = eval_binary_classifier(Y_test,Y_pred)
    models_auc[user].append(metrics["auc_weighted"])
    models_acc[user].append(metrics["balanced_acc"])
    models_eer[user].append(metrics["eer"])

for model in models_auc:
    models_auc_mean[model] = mean(models_auc[model])

for model in models_acc:
    models_acc_mean[model] = mean(models_acc[model]) 

for model in models_eer:
    models_eer_mean[model] = mean(models_eer[model]) 


print("     EER, Balanced Accuracy, Weighted AUC score")
for user in models_auc:
    print("User {}, {}, {}, {}".format(user, round(models_eer_mean[user],4), round(models_acc_mean[user],4), round(models_auc_mean[user],4)))

print("Average over all users:")
lst = []
for model in models_auc_mean:
    lst.append(models_auc_mean[model])
print("Weighted AUC score avg.: " + str(round(mean(lst),4)))

lst=[]

for model in models_acc_mean:
    lst.append(models_acc_mean[model])
print("Balanced Acc avg.: " + str(round(mean(lst),4)))

lst = []

for model in models_eer_mean:
    lst.append(models_eer_mean[model])
print("EER avg.: "+ str(round(mean(lst),4)))
