import json
import pickle
import sys
from datetime import datetime
from json import JSONEncoder
import numpy as np
import pandas as pd
import watchdog.events
import watchdog.observers
import time
import tensorflow as tf
import configparser
import os

from kafka import KafkaProducer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.keras.models import load_model

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import eventsToFeatures
import warnings

warnings.filterwarnings('ignore')

conf_path = './config.config'
max_message_size = 104857600 #bytes
# ndarray json encoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Kafka Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],max_request_size=max_message_size)  # value_serializer=serializer


# Watchdog part for monitoring creation of csv files from mmt
class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Watch for new csvs from mmt probe folder (./server/csv/)
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*_1__data.csv'],
                                                             ## Monitoring csv repot files (_1_) with name with _data
                                                             ignore_directories=True, case_sensitive=False)

    def on_closed(self, event):
        print("Closing action on csv - % s." % event.src_path)
        start_time = time.time()
        mmt_csv = event.src_path
        ips, x_features = eventsToFeatures(mmt_csv)

        # if there are more ips then grouped samples from features (i.e. there is an ip but no features for the ip) -> we delete the ip from ip list
        ips = pd.merge(ips, x_features, how='inner', on=['ip.session_id', 'meta.direction'])
        ips = ips[['ip.session_id', 'meta.direction', 'ip']]
        x_features.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

        print("Prediction - test")
        # rescaling with scaler used with trained model
        x_test = np.asarray(x_features, np.float32)
        x_test = scaler.transform(x_test)

        # prediction
        y_pred = model.predict(x_test)
        y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )

        preds = np.array([y_pred]).T

        # adding predictions to features as last column
        res = np.append(x_features, preds, axis=1)
        res = np.append(ips, res, axis=1)

        # print(res.nbytes)
        # results json encoding
        j_res = json.dumps(res, cls=NumpyArrayEncoder).encode('utf-8')

        print(f'Producing message @ {datetime.now()} | Message')  # = {str(j_res)}')
        psend = producer.send('predictions', j_res)
        # print(psend)
        producer.flush()

        # pd.DataFrame(res).to_csv(f"{predictions_dir}predictions_{classification_id}.csv", index=False,
        #                          header=prediction_names)
        print("--- %s seconds ---" % (time.time() - start_time))
        y_pred = None
        res = None
        features = None


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(conf_path)

    mmt_csv_dir = config['DEFAULT']['mmt_probe_csv_dir']
    model_path = config['DEFAULT']['model_path']
    scaler_path = config['DEFAULT']['scaler_path']
    print(f'{mmt_csv_dir},{model_path},{scaler_path}')
    if not mmt_csv_dir or not model_path or not scaler_path:
        exit('Config does not contain all needed paths')

    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.\nLoading scaler...")
    scaler = pickle.load(open(scaler_path, 'rb'))  # "./saved_scalers/scaler_2022-03-02_10-37-27.pkl"
    print("Scaler loaded.")

    res=np.ndarray(shape=(2,2), dtype=float, order='F')
    j_res = json.dumps(res, cls=NumpyArrayEncoder).encode('utf-8')

    print(f'Producing message @ {datetime.now()} | Message')  # = {str(j_res)}')
    asd = producer.send('messages', j_res)
    # asd = producer.send('messages', 'j_res')
    print(asd)
    producer.flush()

    event_handler = Handler()
    observer = watchdog.observers.Observer()
    print("Starting watchdog.")
    observer.schedule(event_handler, path=mmt_csv_dir, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
