import os 
import matplotlib.pyplot as plt 
from tensorflow import keras 
import numpy as np
from sklearn.metrics import *
import statistics as st


def find_EER(far, frr):
   far_optimum = 0
   frr_optimum = 0
   x = np.absolute((np.array(far) - np.array(frr)))
   y = np.nanargmin(x)
   far_optimum = far[y]
   frr_optimum = frr[y]
   return far_optimum, frr_optimum

def eval_binary_classifier(y_true, y_pred, class_weights=None):
    # NB: `y_true = [1, 0, 0, ...]` but `y_pred = [[0.2324], [0.8731], ...]`.
    # Remember we're dealing with 1 output neuron with sigmoid activation.
    prob_labels = np.array(y_pred).ravel()
    pred_labels = prob_labels > 0.5

    true_labels = y_true.astype(int)
    pred_labels = pred_labels.astype(int)

    # Compute the EER metric
    fpr, tpr,  threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr # get FNR , however FPR is same as FAR
    far_optimum, frr_optimum = find_EER(fpr, fnr)
    EER = max(far_optimum, frr_optimum)

    return {
        'acc': accuracy_score(true_labels, pred_labels),
        'prf_weighted': precision_recall_fscore_support(true_labels, pred_labels, average='weighted'),
        'prf_binary': precision_recall_fscore_support(true_labels, pred_labels, average='binary'),
        'prf_micro': precision_recall_fscore_support(true_labels, pred_labels, average='micro'),
        'prf_macro': precision_recall_fscore_support(true_labels, pred_labels, average='macro'),
        'auc_weighted': roc_auc_score(y_true, y_pred, average='weighted'),
        'auc_micro': roc_auc_score(y_true, y_pred, average='micro'),
        'auc_macro': roc_auc_score(y_true, y_pred, average='macro'),
        'conf_matrix': confusion_matrix(true_labels, pred_labels),
        'eer':EER
    }

base_dir = "balanced-models"

models_auc = {"user7": [], "user9":[], "user12":[], "user15":[],"user16":[],"user20":[], "user21":[], "user23":[],'user29':[],"user35":[]}
models_acc = {"user7": [], "user9":[], "user12":[], "user15":[],"user16":[],"user20":[], "user21":[], "user23":[],'user29':[],"user35":[]}
models_eer ={"user7": [], "user9":[], "user12":[], "user15":[],"user16":[],"user20":[], "user21":[], "user23":[],'user29':[],"user35":[]}
models_auc_mean = {}
models_acc_mean = {}
models_eer_mean = {}

# Evaluation of every trained model to finally compare them
for model_file_name in os.listdir(base_dir):
    loaded_model = keras.models.load_model(os.path.join(base_dir, model_file_name))
    # obtain the testing set from the given model
    X_test = np.load(os.path.join("models-allusers-testingsets", model_file_name[:-3] + "-test-X.npy"))
    Y_test = np.load(os.path.join("models-allusers-testingsets", model_file_name[:-3] + "-test-Y.npy"))

    user = model_file_name.split('-')[0]

    Y_pred = loaded_model.predict(X_test).ravel()
    metrics = eval_binary_classifier(Y_test,Y_pred)
    models_auc[user].append(metrics["auc_macro"])
    models_acc[user].append(metrics["acc"])
    models_eer[user].append(metrics["eer"])

for model in models_auc:
    models_auc_mean[model] = st.mean(models_auc[model])

for model in models_acc:
    models_acc_mean[model] = st.mean(models_acc[model]) 

for model in models_eer:
    models_eer_mean[model] = st.mean(models_eer[model]) 

print(models_auc_mean)
print(models_acc_mean)
print(models_eer_mean)

for user in models_auc:
    print("\hline")
    print("{}&{}&{}&{}\\\\".format(user, round(models_eer_mean[user],4), round(models_acc_mean[user],4), round(models_auc_mean[user],4)))

lst = []
for model in models_auc_mean:
    lst.append(models_auc_mean[model])
print(round(st.mean(lst),4))

lst=[]

for model in models_acc_mean:
    lst.append(models_acc_mean[model])
print(round(st.mean(lst),4))

lst = []

for model in models_eer_mean:
    lst.append(models_eer_mean[model])
print(round(st.mean(lst),4))

