import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import eventsToFeatures
from mmt.FeatureExtractor import feature_names

fn = feature_names.copy()
fn.append('malware')


# python3 tests/run_predictions.py path_to_events_report_from_mmt_probe_config
def main():
    args = sys.argv[1:]
    mmt_reports_dir = args[0]
    # cnn_model_path = args[1]
    output_path = args[1]
    if len(args) <= 2:
        cnn_path = "/home/mra/Documents/Montimage/encrypted-trafic/entra/saved_models/sae_cnn_2022-01-21_15-17-53.h5"
    else:
        cnn_path = args[2]
    cnn = load_model(cnn_path)
    for i in os.listdir(mmt_reports_dir):
        if i.endswith("ot.csv"):
            # if i.endswith('_1_data.csv'):
            start_time = time.time()
            d = datetime.now()
            print("Processing {}".format(i))
            features = eventsToFeatures(str(mmt_reports_dir + i))
            print("Prediction - test")

            y_pred = cnn.predict(features)
            y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )

            preds = np.array([y_pred]).T
            res = np.append(features, preds, axis=1)
            # print(res)
            pd.DataFrame(res).to_csv(output_path + "/predictions_{}.csv".format(d.strftime("%Y-%m-%d_%H-%M-%S")),
                                     index=False, header=fn)
            print("--- %s seconds ---" % (time.time() - start_time))
            y_pred = None
            res = None
            features = None


main()
