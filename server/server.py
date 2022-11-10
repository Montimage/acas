import os.path
import subprocess
import sys
from datetime import datetime
from time import time
from pathlib import Path

import flask
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from flask import Flask, request, make_response, Response

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import eventsToFeatures
from tools.tools import dataScale_cnn
from model.sae_cnn import trainSAE_CNN
from tools.tools import saveConfMatrix, saveScores

"""
    API for training, testing, predicting, described in PUZZLE deliverable 3.5
"""


prediction_names = ['ip.session_id', 'meta.direction', 'ip', 'ip.pkts_per_flow', 'duration', 'ip.header_len',
                    'ip.payload_len', 'ip.avg_bytes_tot_len', 'time_between_pkts_sum',
                    'time_between_pkts_avg', 'time_between_pkts_max',
                    'time_between_pkts_min', 'time_between_pkts_std', '(-0.001, 50.0]',
                    '(50.0, 100.0]', '(100.0, 150.0]', '(150.0, 200.0]', '(200.0, 250.0]',
                    '(250.0, 300.0]', '(300.0, 350.0]', '(350.0, 400.0]', '(400.0, 450.0]',
                    '(450.0, 500.0]', '(500.0, 550.0]', 'tcp_pkts_per_flow', 'pkts_rate',
                    'tcp_bytes_per_flow', 'byte_rate', 'tcp.tcp_session_payload_up_len',
                    'tcp.tcp_session_payload_down_len', '(-0.001, 150.0]',
                    '(150.0, 300.0]', '(300.0, 450.0]', '(450.0, 600.0]', '(600.0, 750.0]',
                    '(750.0, 900.0]', '(900.0, 1050.0]', '(1050.0, 1200.0]',
                    '(1200.0, 1350.0]', '(1350.0, 1500.0]', '(1500.0, 10000.0]', 'tcp.fin',
                    'tcp.syn', 'tcp.rst', 'tcp.psh', 'tcp.ack', 'tcp.urg', 'sport_g', 'sport_le', 'dport_g',
                    'dport_le', 'mean_tcp_pkts', 'std_tcp_pkts', 'min_tcp_pkts',
                    'max_tcp_pkts', 'entropy_tcp_pkts', 'mean_tcp_len', 'std_tcp_len',
                    'min_tcp_len', 'max_tcp_len', 'entropy_tcp_len', 'ssl.tls_version', 'malware']

classification_next_id = 1
model_next_id = 1
app = Flask(__name__)

models_dir = os.path.join('./', 'models')
# print( os.path.join(app.instance_path, 'models'))
# os.makedirs(models_dir, exists_ok=True)

# predictions_dir = os.path.join(app.instance_path, 'csv')
predictions_dir = os.path.join('./server/', 'csv')
# os.makedirs(predictions_dir, exists_ok=True)
print(os.getcwd())

default_model_path = './saved_models/sae_cnn_2022-03-01_10-17-30.h5'
current_path = default_model_path
model = None


def load_def_model():
    global model
    global default_model_path
    print(f"Default model : {default_model_path}")
    model = load_model(default_model_path)


# curl -X PUT 'http://127.0.0.1:5000/model?id=1' OK
@app.route('/model', methods=["PUT"])
def change_model():
    id = request.args.get('id')
    global models_dir
    new_path = f'{models_dir}/{id}.h5'
    print(new_path)
    if os.path.isfile(new_path):
        global model
        global current_path
        current_path = new_path
        model = load_model(current_path)
        return f"Loading OK, current path {current_path}"
    else:
        return Response(f'No such file {new_path}',status=400)

# curl -X GET http://127.0.0.1:5000/training-results/1 -F what='conf' OK 
@app.route('/training-results/<int:id>', methods=["GET"])
def get_training_results(id, what='conf'):

    print("MODELS:"+models_dir)
    
    if id >= model_next_id or id < 1:
        return f"There is no model with id={id}\n"
    if what == 'conf':
        res_dir = f'{models_dir}/conf_matrix_{id}.csv'
    elif what == 'stats':
        res_dir =f'{models_dir}/stats_{id}.csv'
    elif what == 'preds':
        res_dir = f'{models_dir}/test_predictions_{id}.csv'
    else:
        return Response("Wrong option, available only conf/stats/preds",status=400)

    if os.path.isfile(res_dir):
        print(res_dir)
        return flask.send_file(path_or_file=res_dir, mimetype='text/csv', as_attachment=True)
    else:
        return Response(f"Sth wrong with the result file {what}",status=400)

# curl -X GET http://127.0.0.1:5000/model/1 --output 'output.h5' OK
@app.route('/model/<int:id>', methods=["GET"])
def get_model(id):
    m_path = f"{models_dir}/{id}.h5"
    print(m_path)
    if id >= model_next_id or id < 1:
        return Response(f"There is no model with id={id}",status=400)
    else:
        # response = make_response(load_model(m_path))
        # response.headers['Content-Type'] = "application/octet-stream"
        # response.mimetype = 'text/csv'
        return flask.send_file(path_or_file=m_path, mimetype='application', as_attachment=True)

# curl -X POST "http://127.0.0.1:5000/model?nb_epoch_sae=2&batch_size_sae=16&nb_epoch_cnn=1&batch_size_cnn=32" -F "files=@BotTrain_31704_samples.csv" -F "files=@BotTest_13586_samples.csv" OK
@app.route("/model", methods=["POST"])
def train_model():
    files = request.files.getlist('files')
    if len(files) != 2:
        return Response("Failed number of file arguments",status=400)

    nb_epoch_sae = request.args.get('nb_epoch_sae')
    if nb_epoch_sae is None:
        nb_epoch_sae = 10
    else:
        nb_epoch_sae = int(nb_epoch_sae)

    batch_size_sae = request.args.get('batch_size_sae')
    if batch_size_sae is None:
        batch_size_sae = 16
    else:
        batch_size_sae = int(batch_size_sae)

    nb_epoch_cnn = request.args.get('nb_epoch_cnn')
    if nb_epoch_cnn is None:
        nb_epoch_cnn = 5
    else:
        nb_epoch_cnn = int(nb_epoch_cnn)

    batch_size_cnn = request.args.get('batch_size_cnn')
    if batch_size_cnn is None:
        batch_size_cnn = 32
    else:
        batch_size_cnn = int(batch_size_cnn)

    # nb_epoch_sae = 10, batch_size_sae = 16, nb_epoch_cnn = 5, batch_size_cnn = 32
    global model_next_id
    model_id = model_next_id

    train_data = pd.read_csv(files[0])
    test_data = pd.read_csv(files[1])
    # for file in request.files.getlist('files'):
    #     data = pd.read_csv(file)

    d = datetime.now()
    x_train_norm, x_train_mal, x_test_norm, x_test_mal, x_train, y_train, x_test, y_test = dataScale_cnn(train_data, test_data, datetime=d)
    # print(x_train_norm.shape)
    # print(x_train.shape)
    # print(x_train_mal.shape)
    # exec
    input_dim = x_train.shape[1]

    cnn = trainSAE_CNN(x_train_norm=x_train_norm, x_train_mal=x_train_mal,
                       x_train=x_train, y_train=y_train,
                       nb_epoch_cnn=nb_epoch_cnn, nb_epoch_sae=nb_epoch_sae,
                       batch_size_cnn=batch_size_cnn, batch_size_sae=batch_size_sae, datenow=d)

    cnn.save(f'{models_dir}/{model_id}.h5')

    print("Prediction - test")
    y_pred = cnn.predict(x_test)
    y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
    # print("Metrics")

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    pd.DataFrame(cm).to_csv(f'{models_dir}/conf_matrix_{model_id}.csv')
    saveScores(y_true=y_test, y_pred=y_pred, filepath=f'{models_dir}/stats_{model_id}.csv')

    preds = np.array([y_pred]).T
    res = np.append(x_test, preds, axis=1)
    pd.DataFrame(res).to_csv(f'{models_dir}/test_predictions_{model_id}.csv', index=False, header=test_data.columns)

    # flask.jsonify(data)
    # response = make_response(pd.DataFrame(res).to_csv())
    # response.headers['Content-Type'] = "application/octet-stream"
    # response.mimetype = 'text/csv'

    model_next_id += 1
    # print(model_next_id)
    return Response(f"\n\nTraining/testing is done, new model id is: {model_id}",status=200)

# curl -X GET http://127.0.0.1:5000/classification/1
@app.route('/classification/<int:id>', methods=["GET"])
def get_classification(id):
    result = f"{predictions_dir}/predictions_{id}.csv"
    if id >= classification_next_id or id < 1:
        return Response(f"There is no classification with id={id}",status=400)
    elif os.path.isfile(result):
        with open(result, "r") as f:
            return flask.jsonify(f.read())
            # return f.read()
    else:
        return Response(f"The predictions_{id}.csv file is not ready",status=425)

# curl -X POST --data-binary "@ex.cap" http://127.0.0.1:5000/classification
@app.route("/classification", methods=["POST"])
def add_classification():
    global classification_next_id
    classification_id = classification_next_id
    global predictions_dir
    # csv_path = "./csv/"

    # if not os.path.isfile(f"pcap/{classification_id}.pcap"):
    #     return Response(f"No pcap file",status=400)
    print(os.getcwd())
    with open(f"./server/pcap/{classification_id}.pcap", "wb") as f:
        f.write(request.get_data())
        f.close()

    # Run probe on the pcap file
    # Example: ./probe -X input.source="./pcap/3.pcap" -X file-output.output-file="3.csv"
    #           -X file-output.output-dir="./csv/"
    print(os.getcwd())
    subprocess.call(["./server/probe",
                     "-c", f'./server/mmt-probe.conf',
                     "-X", f'input.source=./server/pcap/{classification_id}.pcap',
                     "-X", f'file-output.output-file={classification_id}.csv',
                     "-X", f'file-output.output-dir={predictions_dir}/'])

    for filename in Path(predictions_dir).glob(f"*_0_{classification_id}.csv"):
        filename.unlink()

    # cnn_path = "/home/mra/Documents/Montimage/encrypted-trafic/entra/saved_models/sae_cnn_2022-02-02_16-15-25.h5"
    # cnn = load_model(current_path)

    for filename in Path(predictions_dir).glob(f"*_1_{classification_id}.csv"):
        filename.rename(f"./server/csv/{classification_id}.csv")

    input_csv = f"./server/csv/{classification_id}.csv"

    start_time = time()
    print("Processing {}".format(input_csv))
    ips, features = eventsToFeatures(input_csv)

    # if there are more ips then grouped samples from features (i.e. there is an ip but no features for the ip) -> we delete the ip from ip list
    ips = pd.merge(ips, features, how='inner', on=['ip.session_id', 'meta.direction'])
    ips = ips[['ip.session_id', 'meta.direction', 'ip']]
    features.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

    print("Prediction - test")
    #
    global model
    y_pred = model.predict(features)
    y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )

    preds = np.array([y_pred]).T

    res = np.append(features, preds, axis=1)
    res = np.append(ips, res, axis=1)

    # print(res)
    pd.DataFrame(res).to_csv(f"{predictions_dir}/predictions_{classification_id}.csv", index=False, header=prediction_names)
    print("--- %s seconds ---" % (time() - start_time))
    y_pred = None
    res = None
    features = None

    classification_next_id += 1
    return Response(f"\n\nClassification is done, classification id: {classification_id}",status=200)


if __name__ == "__main__":
    print(("Please wait until server has fully started"))
    load_def_model()
    app.run()
