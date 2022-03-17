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


def createTrainTestSet(path, train_samples, test_samples):
    """

    :param path:
    :param train_samples:
    :param test_samples:
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

# runMMT(pcap_dir='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu_bot_1/output_00000_19700101010006_filtered.pcap',
#        csv_dir='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu_bot_1/',
#        csv_name='malicious')
# pickleFeatureFilesFromFile(in_file="/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu_bot_1/malicious.csv",
#                            out_file="/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu_bot_1/malicious",
#                            is_malware=True)
#################
# runMMT(pcap_dir='./data/ctu/2013-12-17_capture1.pcap', csv_dir='./data/ctu/', csv_name='normal')
# runMMT(pcap_dir='./data/ctu/attacker-10.0.0.42.pcap', csv_dir='./data/ctu/', csv_name='attacker')
# createSetFromCSV(in_file_normal='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/normal.csv',
#                  out_file_normal='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/normal',
#                  in_file_mal='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/attacker.csv',
#                  out_file_mal='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/attacker',
#                  train_test_path='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/',
#                  train_samples_no=int(n_of_samples_total*0.3),
#                  test_samples_no=int(n_of_samples_total*0.7))

#################
# pickleFeatureFilesFromFile("/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/attacker.csv",
#                            "/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/attacker",
#                            is_malware=True)
# pickleFeatureFilesFromFile("/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/normal.csv",
#                            "/home/mra/Documents/Montimage/encrypted-trafic/entra/data/ctu/normal",
#                            is_malware=False)
# print("Train/test set.")
# createTrainTestSet(path='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mixed-2/',
#                      train_samples=int(n_of_samples_total * 0.7),
#                      test_samples=int(n_of_samples_total * 0.3))
# print("Train/test set completed.")
