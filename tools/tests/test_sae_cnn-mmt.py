from datetime import datetime

import seaborn as sn
import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, \
    f1_score, classification_report
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys


sys.path.append(sys.path[0] + '/..')
# from models.sae_cnn import trainCNN
# from tools.tools import CIC_col_names, CIC_types_dict, saveConfMatrix, saveScores
from tools.tools import saveConfMatrix, saveScores, dataScale_cnn
from models.sae_cnn_total import trainSAE_CNN


train_data = pd.read_csv(
    './data/cic-mmt/BotTrain_31704_samples.csv', delimiter=",")
train_data.drop(columns=['ip.session_id','meta.direction'], inplace=True)
# train_data.drop(columns=train_data.columns[0], axis=1,inplace=True)
    # './data/CIC/Train_BalancedSet_1000000_samples.csv', delimiter=",", names=CIC_col_names, dtype=CIC_types_dict, skiprows=1)
    # nrows=500000)
test_data = pd.read_csv(
    './data/cic-mmt/BotTest_13586_samples.csv', delimiter=",")
test_data.drop(columns=['ip.session_id','meta.direction'], inplace=True)
# test_data.drop(columns='meta.direction', inplace=True)
# test_data.drop(columns=test_data.columns[0], axis=1,inplace=True)
# './data/CIC/Test_BalancedSet_100000_samples.csv', delimiter=",", names=CIC_col_names, dtype=CIC_types_dict, skiprows=1)#,
    # nrows=10000)

x_train_norm, x_train_mal, x_test_norm, x_test_mal, x_train, y_train, x_test, y_test = dataScale_cnn(train_data, test_data)

# exec
input_dim = x_train.shape[1]
nb_epoch_sae = 10 # 30#10000
batch_size_sae = 16#128
nb_epoch_cnn = 5
batch_size_cnn = 32

d = datetime.now()
cnn = trainSAE_CNN(x_train_norm=x_train_norm, x_train_mal=x_train_mal,
               x_train=x_train, y_train=y_train,
               nb_epoch_cnn=nb_epoch_cnn,nb_epoch_sae=nb_epoch_sae,
               batch_size_cnn=batch_size_cnn, batch_size_sae=batch_size_sae, datenow=d)

print("Prediction - test")
y_pred = cnn.predict(x_test)
y_pred = numpy.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
print("Metrics")


print(classification_report(y_test, y_pred))
saveConfMatrix(y_true=y_test, y_pred=y_pred,
               filepath_csv='results/sae-cnn/conf_matrix_sae_test-cnn_{}.csv'.format(d.strftime("%Y-%m-%d_%H-%M-%S")),
               filepath_png='results/sae-cnn/conf_matrix_sae-cnn_{}.jpg'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
saveScores(y_true=y_test, y_pred=y_pred, filepath='results/sae-cnn/stats_{}'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))


preds = np.array([y_pred]).T
res = np.append(x_test,preds,axis=1)
pd.DataFrame(res).to_csv("results/sae-cnn/predictions_{}.csv".format(d.strftime("%Y-%m-%d_%H-%M-%S")), index=False, header=test_data.columns)

# print("Prediction - validation")
#
# y_pred = cnn.predict(x_val)
# y_pred = numpy.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
# print("Metrics")
#
#
# print(classification_report(y_val, y_pred))
# saveConfMatrix(y_true=y_val, y_pred=y_pred,
#                filepath_csv='results/sae-cnn/conf_matrix_sae_val-cnn_{}.csv'.format(d.strftime("%Y-%m-%d_%H-%M-%S")),
#                filepath_png='results/sae-cnn/conf_matrix_sae-cnn_{}.jpg'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
# saveScores(y_true=y_val, y_pred=y_pred, filepath='results/sae-cnn/stats_{}'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))

# y_pred
# cm = confusion_matrix(y_test, y_pred)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# df_cfm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
# plt.figure(figsize=(15, 12))
# cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='.1f')
# d = datetime.now()
# cfm_plot.figure.savefig('conf_matrix {}.jpg'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
