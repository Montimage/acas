import os
import sys
import numpy as np
import pandas as pd
import constants
from tensorflow.keras.models import load_model
from eventToFeature import eventsToFeatures

sys.path.append(sys.path[0] + '/..')

def predict(csv_path, model_path, result_path):
    ips, features = eventsToFeatures(csv_path)
    if len(ips) == 0:
        print('There is no ip traffic to predict')
        return
    # if there are more ips then grouped samples from features (i.e. there is an ip but no features for the ip) -> we delete the ip from ip list
    print("Going to merge features if there are more ips")
    ips = pd.merge(ips, features, how='inner', on=['ip.session_id', 'meta.direction'])
    ips = ips[['ip.session_id', 'meta.direction', 'ip']]
    features.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)

    print("Going to test the prediction")
    model = load_model(model_path)
    print("Model has been loaded from")
    
    # Check if feature dimensions match the model's expected input
    expected_features = model.input_shape[1]
    current_features = features.shape[1]
    
    if current_features != expected_features:
        print(f"Warning: Feature mismatch - Model expects {expected_features} features, but got {current_features}")
        
        if current_features > expected_features:
            # Too many features - select the first N features
            print(f"Selecting first {expected_features} features to match model input")
            features = features.iloc[:, :expected_features]
        else:
            # Too few features - pad with zeros
            print(f"Padding with {expected_features - current_features} zero columns")
            padding = pd.DataFrame(np.zeros((features.shape[0], expected_features - current_features)))
            features = pd.concat([features, padding], axis=1)
    
    y_pred = model.predict(features)
    y_pred = np.transpose(np.round(y_pred)).reshape(y_pred.shape[0], )
    preds = np.array([y_pred]).T
    nb_attacks = np.count_nonzero(preds != 0)
    res = np.append(features, preds, axis=1)
    res = np.append(ips, res, axis=1)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    dataFrame = pd.DataFrame(res)
    print("Total flows: "+ str(len(dataFrame.index)))
    
    # Use only the header values that match the actual DataFrame columns
    num_cols = len(dataFrame.columns)
    header = constants.AD_FEATURES_OUTPUT[:num_cols] if num_cols <= len(constants.AD_FEATURES_OUTPUT) else None
    last_column_index = num_cols - 1
    
    # Determine if files exist (for header writing)
    predictions_exists = os.path.exists(f"{result_path}/predictions.csv")
    attacks_exists = os.path.exists(f"{result_path}/attacks.csv")
    normals_exists = os.path.exists(f"{result_path}/normals.csv")
    stats_exists = os.path.exists(f"{result_path}/stats.csv")
    
    # Append to predictions.csv (write header only if file doesn't exist)
    dataFrame.to_csv(f"{result_path}/predictions.csv", mode='a', index=False, header=header if not predictions_exists else False)

    attackDF = dataFrame[dataFrame[last_column_index] > 0]
    print("Number of attacks: " + str(len(attackDF.index)))
    # Append to attacks.csv (write header only if file doesn't exist)
    if len(attackDF.index) > 0:
        attackDF.to_csv(f"{result_path}/attacks.csv", mode='a', index=False, header=header if not attacks_exists else False)

    normalDF = dataFrame[dataFrame[last_column_index] == 0]
    print("Number of normals: " + str(len(normalDF.index)))
    # Append to normals.csv (write header only if file doesn't exist)
    if len(normalDF.index) > 0:
        normalDF.to_csv(f"{result_path}/normals.csv", mode='a', index=False, header=header if not normals_exists else False)

    # For stats.csv, we want cumulative totals
    # Read existing stats if present, otherwise start from zero
    cumulative_normal = len(normalDF.index)
    cumulative_attack = len(attackDF.index)
    cumulative_total = len(dataFrame.index)
    
    if stats_exists:
        try:
            existing_stats = pd.read_csv(f"{result_path}/stats.csv", header=None)
            if len(existing_stats) > 0:
                last_row = existing_stats.iloc[-1]
                cumulative_normal += int(last_row[0])
                cumulative_attack += int(last_row[1])
                cumulative_total += int(last_row[2])
        except Exception as e:
            print(f"Warning: Could not read existing stats: {e}")
    
    # Append cumulative stats
    statsArray = np.array([[cumulative_normal, cumulative_attack, cumulative_total]])
    pd.DataFrame(statsArray).to_csv(f"{result_path}/stats.csv", mode='a', index=False, header=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print('Invalid inputs')
    else:
        csv_path = sys.argv[1]
        model_path = sys.argv[2]
        result_path = sys.argv[3]
        predict(csv_path, model_path, result_path)