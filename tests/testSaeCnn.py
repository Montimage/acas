import pickle
from datetime import datetime
import numpy
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import sys

sys.path.append(sys.path[0] + '/..')
from tools.tools import saveConfMatrix, saveScores, dataScale_cnn
from model.sae_cnn import trainSAE_CNN

"""
    Runs tests (pretty much the same as runPredictions)

"""

train_data = pd.read_csv(
    './data/BotTrain_31704_samples.csv', delimiter=",")

train_data.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

test_data = pd.read_csv('./data/BotTest_13586_samples.csv', delimiter=",")
test_data.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

d = datetime.now()
x_train_norm, x_train_mal, x_test_norm, x_test_mal, x_train, y_train, x_test, y_test, scaler = dataScale_cnn(train_data,
                                                                                                             test_data, datetime=d)

# exec
input_dim = x_train.shape[1]
nb_epoch_sae = 5  # 30#10000
batch_size_sae = 16  # 128
nb_epoch_cnn = 2
batch_size_cnn = 32

cnn = trainSAE_CNN(x_train_norm=x_train_norm, x_train_mal=x_train_mal,
                   x_train=x_train, y_train=y_train,
                   nb_epoch_cnn=nb_epoch_cnn, nb_epoch_sae=nb_epoch_sae,
                   batch_size_cnn=batch_size_cnn, batch_size_sae=batch_size_sae, datenow=d)

print("Prediction - test")
y_pred = cnn.predict(x_test)
# print(y_pred)
y_pred = numpy.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )

print("Metrics")
# print(y_pred)
# print(y_test)

print(classification_report(y_test, y_pred))
saveConfMatrix(y_true=y_test, y_pred=y_pred,
               filepath_csv='./saved_results/sae-cnn/conf_matrix_sae_test-cnn_{}.csv'.format(d.strftime("%Y-%m-%d_%H-%M-%S")),
               filepath_png='./saved_results/sae-cnn/conf_matrix_sae-cnn_{}.jpg'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))
saveScores(y_true=y_test, y_pred=y_pred, filepath='./saved_results/sae-cnn/stats_{}'.format(d.strftime("%Y-%m-%d_%H-%M-%S")))

preds = np.array([y_pred]).T
res = np.append(x_test, preds, axis=1)
pd.DataFrame(res).to_csv("./saved_results/sae-cnn/predictions_{}.csv".format(d.strftime("%Y-%m-%d_%H-%M-%S")), index=False,
                         header=test_data.columns)
