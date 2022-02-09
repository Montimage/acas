import os.path
import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')
from mmt.FeatureExtractor import calculateFeatures

tls_event = ["ssl.tls_version",
             "ssl.tls_content_type",
             "ssl.tls_length",
             "ssl.ssl_handshake_type",
             "ip.session_id",
             "meta.direction"]

tcp_event = ["tcp.src_port",
             "tcp.dest_port",
             "tcp.payload_len",
             "tcp.fin",
             "tcp.syn",
             "tcp.rst",
             "tcp.psh",
             "tcp.ack",
             "tcp.urg",
             "tcp.tcp_session_payload_up_len",
             "tcp.tcp_session_payload_down_len",
             "ip.session_id",
             "meta.direction"]  # if = 0 then its client -> server (Checked with syn=1 ack=0)

ipv4_event = ["time",
              "ip.version",
              "ip.session_id",
              "meta.direction",
              "ip.first_packet_time",
              "ip.last_packet_time",
              "ip.header_len",
              "ip.tot_len",
              "ip.src",
              "ip.dst"
              ]

ipv6_event = ["time",
              "ip.version",
              "ip.session_id",
              "meta.direction",
              "ip.first_packet_time",
              "ip.last_packet_time",
              "ip.src",
              "ip.dst"
              ]


#    handshake_type codes: total = 24
# any_event = ["time", "meta.source_id"]

def readMMTcsv(path):
    with open(path, 'r') as temp:
        col_count = [len(l.split(",")) for l in temp.readlines()]
        col_names = [i for i in range(0, max(col_count))]
        types_dict = {'0': int, '1': int, '2': "string", '3': 'float', '4': 'string'}
        types_dict.update({col: str for col in col_names if col not in types_dict})
        df = pd.read_csv(path, header=None, delimiter=",", names=col_names, dtype=types_dict)
        print("Read {} lines".format(df.shape[0]))
        return df


def extractReport(df, report_name):
    new_df = df[df[4] == report_name].copy()
    # if report_name == "any-event":
    #     if not new_df.empty:
    #         new_df.columns = any_event  ##colnames from mmt-probe conf file
    #     else:
    #         new_df = pd.DataFrame(columns=any_event)
    if report_name == "ipv4-event":
        new_df.drop(columns=[0, 1, 2, 4],
                    inplace=True)  # ignoring 3 first columns (report id, ?, filepath) and 5th (report name)
        new_df.dropna(axis=1, inplace=True)
        # print(new_df)
        if not new_df.empty:
            new_df.columns = ipv4_event  ##colnames from mmt-probe conf file
        else:
            new_df = pd.DataFrame(columns=ipv4_event)

    elif report_name == "ipv6-event":
        new_df.drop(columns=[0, 1, 2, 4],
                    inplace=True)  # ignoring 3 first columns (report id, ?, filepath) and 5th (report name)
        new_df.dropna(axis=1, inplace=True)
        # print(new_df)
        if not new_df.empty:
            new_df.columns = ipv6_event  ##colnames from mmt-probe conf file
        else:
            new_df = pd.DataFrame(columns=ipv6_event)

    else:
        new_df.drop(columns=[0, 1, 2, 3, 4],
                    inplace=True)  # ignoring 3 first columns (report id, ?, time, filepath) and 5th (report name)
        new_df.dropna(axis=1, inplace=True)
        if report_name == "tcp-event":
            if not new_df.empty:
                new_df.columns = tcp_event  ##colnames from mmt-probe conf file
            else:
                new_df = pd.DataFrame(columns=tcp_event)
        elif report_name == "tls-event":
            if not new_df.empty:
                new_df.columns = tls_event  ##colnames from mmt-probe conf file
            else:
                new_df = pd.DataFrame(columns=tls_event)
    return new_df


def readAndExtractEvents(path):
    df = readMMTcsv(path)

    # any_traffic = p1.extractReport("any-event")
    ipv4_traffic = extractReport(df, "ipv4-event")
    ipv6_traffic = extractReport(df, "ipv6-event")
    ip_traffic = ipv4_traffic.append(ipv6_traffic, sort=False)
    ip_traffic = ip_traffic.replace(np.nan, 0)
    tcp_traffic = extractReport(df, "tcp-event")
    tls_traffic = extractReport(df, "tls-event")
    df = df[0:0]

    return ip_traffic, tcp_traffic, tls_traffic


def eventsToFeatures(in_csv):
        ip_traffic, tcp_traffic, tls_traffic = readAndExtractEvents(in_csv)
        ips, p1_features = calculateFeatures(ip_traffic, tcp_traffic, tls_traffic)
        p1_features = p1_features.fillna(0)
        print("Extracted {} features".format(p1_features.shape[0]))
        return ips, p1_features


def trafficToFeatures(in_csv, out_pkl, is_malware=False):
    """

    :param in_csv:
    :param out_pkl:
    :param is_malware:
    :return:
    """
    if not os.path.isfile(out_pkl):
        ip_traffic, tcp_traffic, tls_traffic = readAndExtractEvents(in_csv)
        _, p1_features = calculateFeatures(ip_traffic, tcp_traffic, tls_traffic)
        p1_features = p1_features.fillna(0)
        if is_malware:
            p1_features['malware'] = 1
        else:
            p1_features['malware'] = 0

        print(p1_features.columns)
        p1_features.to_pickle(out_pkl)
        print("Extracted {} features".format(p1_features.shape[0]))
        p1_features = p1_features[0:0]
    else:
        print("Pkl {} already exists".format(out_pkl))


def pickleAllFeatureFilesFromDir(in_path, out_path, is_malware=False):
    for i in os.listdir(in_path):
        print("Processing {}".format(i))
        trafficToFeatures(str(in_path + i), str(out_path + i + '.pkl'), is_malware)


def pickleFeatureFilesFromFile(in_file, out_file, is_malware=False):
    print("Processing {}".format(in_file))
    trafficToFeatures(str(in_file), str(out_file + '.pkl'), is_malware)


def loadAllFeatureFilesFromPickleDir(path, max=1000):
    features = pd.DataFrame()
    c = 0
    for i in os.listdir(path):
        if c >= max:
            return features
        else:
            features = features.append(pd.read_pickle(str(path + i)))
    return features

def loadFeaturesFromPickleFile(pathfile):
    features = pd.DataFrame()
    features = features.append(pd.read_pickle(str(pathfile)))
    return features


# getFeatureFilesFromFile(in_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt/normal.csv',
# out_file='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt-pkl/normal', is_malware=True)
#
# asd = loadFeaturesFromFile(pathfile='/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt-pkl/pkl/bot.pkl')
# asd['malware']=1
# asd.to_pickle('/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt-pkl/pkl/bot.pkl')
# asd.to_csv('/home/mra/Documents/Montimage/encrypted-trafic/entra/data/cic-mmt-pkl/bot_features.csv')

# in_csv = '/home/mra/Documents/Montimage/encrypted-trafic/entra/data/no_tls.csv'
# ip_traffic, tcp_traffic, tls_traffic = readAndExtract(in_csv)

# ip_pkts_per_flow = ip_traffic.groupby(['ip.session_id', 'meta.direction'])[['time']].count().reset_index()
# total_in_packets = in_out_total_packets[in_out_total_packets['meta.direction'] == '0'][['time']]
# total_out_packets = in_out_total_packets[in_out_total_packets['meta.direction'] == '1'][['time']]
# content type ==22 -> handshake --> handshake type is correct
# &&&
# tls_traffic['ssl.tls_content_type'] = pd.to_numeric(tls_traffic['ssl.tls_content_type'])
# tls_handshake_types = tls_traffic[tls_traffic['ssl.tls_content_type'] == 22][
#     ['ssl.tls_content_type', 'ssl.ssl_handshake_type']]
#
# prob_tls_handshake = \
#     tls_traffic[tls_traffic['ssl.tls_content_type'] == 22].groupby(["ip.session_id", "meta.direction"])[
#         'ssl.tls_content_type'].agg(['unique']).count() / 24
# &&&
