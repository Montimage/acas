import os
import pickle
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import eventsToFeatures
from mmt.featureExtractor import feature_names

fn = feature_names.copy()
fn.append('malware')

"""
Runs predictions using .csv reports from mmt_reports_dir (1st arg), saving results in output_path (2nd arg), using
 given model (3rd arg) and given scaler (4th arg)
Execution: 
python3 tests/runPredictions.py path_to_events_report_from_mmt_probe_config output_path path_to_model path_to_scaler

"""

def main():
    args = sys.argv[1:]
    mmt_reports_dir = args[0]
    output_path = args[1]

    if len(args) <= 2:
        cnn_path = "./saved_models/sae_cnn_2022-03-02_10-37-27.h5"
        scaler_path = "./saved_scalers/scaler_2022-03-02_10-37-27.pkl"
        print(f"Model/scaler path not specified, using {cnn_path}, {scaler_path}")
    else:
        cnn_model_path = args[2]
        scaler_path = args[3]

    model = load_model(cnn_path)
    scaler = pickle.load(open(scaler_path, 'rb'))

    for i in os.listdir(mmt_reports_dir):
        if i.endswith("_1_data.csv"):
            start_time = time.time()
            d = datetime.now()
            print("Processing {}".format(i))
            _, features = eventsToFeatures(str(mmt_reports_dir + i))

            features.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

            x_test = np.asarray(features, np.float32)
            x_test = scaler.transform(x_test)

            print("Prediction")
            y_pred = model.predict(x_test)
            y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
            preds = np.array([y_pred]).T
            res = np.append(features, preds, axis=1)
            pd.DataFrame(res).to_csv(output_path + "/predictions_{}.csv".format(d.strftime("%Y-%m-%d_%H-%M-%S")),
                                     index=False, header=fn)
            print("--- %s seconds ---" % (time.time() - start_time))
            y_pred = None
            res = None
            features = None

main()
