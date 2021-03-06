import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import pickleFeatureFilesFromFile


"""
Functions creating training and testing datasets
"""

def createTrainTestSet(path, train_samples, test_samples):
    """
    Creates a training and testing .csv files with balanced 0/1 classes
    :param path: folder with .pkl files (already calculated dataframes with ML features)
    :param train_samples: number of train samples
    :param test_samples: number of test samples
    :return:
    """
    m_ndt = pd.DataFrame()
    no_ndt = pd.DataFrame()
    norm_rest = int((train_samples + test_samples) / 2)
    mal_rest = int((train_samples + test_samples) / 2)
    for i in os.listdir(path):
        if i.endswith('.pkl'):
            if mal_rest > 0 or norm_rest > 0:
                # Reading next file + cleaning data
                print("Processing {}".format(i))
                data = pd.read_pickle(str(path + i))
                print("Data samples before cleaning: " + str(len(data)))
                data = data[np.isfinite(data).all(1)]  # get rid of inf values
                print("Data samples after cleaning: " + str(len(data)))
                if mal_rest > 0:
                    mal = data.loc[data['malware'].isin([1, 2])]  # == 1 or data['malware'] == 2]
                    mal_nb = mal.shape[0]
                    print("Malicious samples: {}".format(mal_nb))

                    mal = shuffle(mal)
                    if mal_nb < mal_rest:
                        mal_samples = mal.sample(n=mal_nb)
                    else:
                        mal_samples = mal.sample(mal_rest)
                    m_ndt = m_ndt.append(mal_samples)
                    mal_rest -= mal_samples.shape[0]

                    print("Added malicious: {}".format(mal_samples.shape[0]))

                if norm_rest > 0:
                    norm = data.loc[data['malware'] == 0]
                    norm_nb = norm.shape[0]
                    print("Normal samples: {}".format(norm.shape[0]))
                    norm = shuffle(norm)
                    if norm_nb < norm_rest:
                        norm_samples = norm.sample(n=norm_nb)
                    else:
                        norm_samples = norm.sample(n=norm_rest)
                    no_ndt = no_ndt.append(norm_samples)
                    norm_rest -= norm_samples.shape[0]
                    print("Added normal: {}".format(norm_samples.shape[0]))

    m_ndt = shuffle(m_ndt)
    no_ndt = shuffle(no_ndt)
    print("malicious total: {}".format(m_ndt.shape[0]))
    print("normal total: {}".format(no_ndt.shape[0]))

    halfset_idx = math.ceil(train_samples / 2)
    train = m_ndt[0:halfset_idx]
    train = train.append(no_ndt[0:halfset_idx])
    train = shuffle(train)

    test = m_ndt[halfset_idx:]
    print(str(test.shape[0]))
    test = test.append(no_ndt[halfset_idx:])
    print(str(test.shape[0]))
    test = shuffle(test)
    print(str(test.shape[0]))
    print("train total: {}".format(train.shape[0]))
    print("test total: {}".format(test.shape[0]))

    train = train.replace(np.nan, 0)
    test = test.replace(np.nan, 0)
    train.to_csv(str(path) + "Train_" + str(train.shape[0]) + "_samples.csv", index=False)
    test.to_csv(str(path) + 'Test_' + str(test.shape[0]) + "_samples.csv", index=False)


def createSetFromCSV(in_file_normal, out_file_normal, in_file_mal, out_file_mal, train_test_path, train_samples_no,
                     test_samples_no):
    '''
    Creates training and testing datasets from MMT csv report files (done from pcaps)

    :param in_file_normal: mmt-probe .csv report file based on normal traffic only
    :param out_file_normal: path for pkl file with calculated ML features (normal traffic only)
    :param in_file_mal: mmt-probe .csv report file based on malicious traffic only
    :param out_file_mal: path for pkl file with calculated ML features (malicious traffic only)
    :param train_test_path: path of data form which train/test set should be created (so should be same as .pkl files) ##TODO
    :param train_samples_no: number of samples to be in training set
    :param test_samples_no: number of samples to be in test set
    :return:
    '''
    ### Creating from MMT csv files (done from pcaps) -> pkl files with features (separate: normal + bot)
    # normal
    pickleFeatureFilesFromFile(in_file_normal, out_file_normal, is_malware=False)
    # malicious
    pickleFeatureFilesFromFile(in_file_mal, out_file_mal, is_malware=True)

    ## Creating from pkl feature files (separated into normal/bot) -> csv divided into train/test sets
    print("Train/test set.")
    createTrainTestSet(path=train_test_path, train_samples=train_samples_no, test_samples=test_samples_no)
    print("Train/test set completed.")


def runMMT(pcap_dir, csv_dir, csv_name):
    '''
    Runs mmt as subprocess
    :param pcap_dir: .pcap to be processed
    :param csv_dir: .csv path of mmt-probe report
    :param csv_name: .csv name of mmt-probe report
    :return:
    '''
    subprocess.call(["./server/probe",
                     "-c", f'./server/mmt-probe.conf',
                     "-X", f'input.source={pcap_dir}',
                     "-X", f'file-output.output-file={csv_name}.csv',
                     "-X", f'file-output.output-dir={csv_dir}'])

    for filename in Path(csv_dir).glob(f"*_0_{csv_name}.csv"):
        filename.unlink()

    for filename in Path(csv_dir).glob(f"*_1_{csv_name}.csv"):
        filename.rename(f"{csv_dir}/{csv_name}.csv")


def createSetFromPcap(pcap_normal_path, csv_normal_name_output, csv_normal_output_dir,
                      pcap_malicious_path, csv_mal_name_output, csv_mal_output_dir,
                      train_test_path,
                      train_samples_no, test_samples_no):
    '''
    Creates training and testing datasets from malicious and normal pcaps

    :param pcap_normal_path: pcap with normal traffic only
    :param csv_normal_name_output: name of .csv of mmt-probe report (normal)
    :param csv_normal_output_dir: .csv of mmt-probe report (normal)
    :param pcap_malicious_path: pcap with malicious traffic only
    :param csv_mal_name_output: name of .csv of mmt-probe report (malicious)
    :param csv_mal_output_dir: .csv of mmt-probe report (malicious)
    :param train_test_path: path of data form which train/test set should be created (so should be same as .pkl files) ##TODO
    :param train_samples_no: number of samples to be in training set
    :param test_samples_no: number of samples to be in test set
    :return:
    '''
    # normal
    runMMT(pcap_dir=pcap_normal_path, csv_dir=csv_normal_output_dir, csv_name=csv_normal_name_output)
    pickleFeatureFilesFromFile(f'{csv_normal_output_dir}/{csv_normal_name_output}.csv',
                               f'{csv_normal_output_dir}/{csv_normal_name_output}', is_malware=False)

    # malicious
    runMMT(pcap_dir=pcap_malicious_path, csv_dir=csv_mal_output_dir, csv_name=csv_mal_name_output)
    pickleFeatureFilesFromFile(f'{csv_mal_output_dir}/{csv_mal_name_output}.csv',
                               f'{csv_mal_output_dir}/{csv_mal_name_output}', is_malware=True)

    ## Creating from pkl feature files (separated into normal/bot) -> csv divided into train/test sets
    print("Train/test set.")
    createTrainTestSet(path=train_test_path, train_samples=train_samples_no, test_samples=test_samples_no)
    print("Train/test set completed.")


# mal 64Mb --> 593 K lines csv MMT --> 46752 samples
# nomal 112Mb --> 362K lines csv MMT --> 21888 samples
# n_of_samples_total = 46752*2  # 478726#572 380  # 286190*2  22646

# n_of_samples_total = 46752*2 + 22646*2   # 478726#572 380  # 286190*2  22646
# n_of_samples_total = 1757 * 2  # 478726#572 380  # 286190*2  22646
n_of_samples_total = 57256
createSetFromPcap(pcap_normal_path='./data/ctu_bot_1/normal/normal2.pcap', csv_normal_name_output='normal2',
                  csv_normal_output_dir='./data/ctu_bot_1/',
                  pcap_malicious_path='./data/ctu_bot_1/output_00004_19700125033810_filtered.pcap', csv_mal_name_output='attacker',
                  csv_mal_output_dir='./data/ctu_bot_1/',
                  train_test_path='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu_bot_1/',
                  train_samples_no=int(n_of_samples_total * 0.7),
                  test_samples_no=int(n_of_samples_total * 0.3))
