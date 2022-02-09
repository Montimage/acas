import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

sys.path.append(sys.path[0] + '/..')
from mmt.readerMMT import pickleFeatureFilesFromFile


def createBotSet(path, train_samples, test_samples):
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
                    mal = data.loc[data['malware'] == 1]
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
    train.to_csv(str(path) + "BotTrain_" + str(train.shape[0]) + "_samples.csv", index=False)
    test.to_csv(str(path) + 'BotTest_' + str(test.shape[0]) + "_samples.csv", index=False)



### Creating from MMT csv files (done from pcaps) -> pkl files with features (separate: normal + bot)
pickleFeatureFilesFromFile(in_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/normal.csv',
                           out_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/normal',
                           is_malware=False)
# pickleFeatureFilesFromFile(in_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/bot.csv',
#                            out_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/bot',
#                            is_malware=True)
#
## Creating from pkl feature files (separated into normal/bot) -> csv divided into train/test sets
print("Bot set.")
n_of_samples_total = 45292  # 478726#572 380  # 286190*2  22646
createBotSet(path='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/',
             train_samples=int(n_of_samples_total * 0.7), test_samples=int(n_of_samples_total * 0.3))
print("Bot set completed.")
