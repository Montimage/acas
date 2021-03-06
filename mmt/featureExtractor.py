import numpy
import pandas as pd
from scipy.stats import entropy


"""
Deals with calculation of actual ML features. 

feature_names - predefined col names for the final ML feature dataframe 

"""

feature_names = ['ip.pkts_per_flow', 'duration', 'ip.header_len',
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
                 'min_tcp_len', 'max_tcp_len', 'entropy_tcp_len', 'ssl.tls_version']


def calculateFeatures(ip_traffic, tcp_traffic, tls_traffic):
    """
    Calculates ML features based on traffic extracted from mmt-probe .csv. Features are calculated per flow and direction
    where direction is identified by mmt-probe. Remark: features are calculated and returned including the direction
    and session id, both columns should be dropped before feeding them into ML model

    :param ip_traffic:
    :param tcp_traffic:
    :param tls_traffic:
    :return: Ip of flows and dataframe with ML features (per flow+direction)
    """

    print("Extracting features")

    # Bins of packet lengths and time between packets based on
    # "MalDetect: A Structure of Encrypted Malware Traffic Detection" by Jiyuan Liu et al.

    bins_len = [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 10000]
    bins_time = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

    ## saving unique ips based on ip_traffic
    ips = ip_traffic.groupby(["ip.session_id", "meta.direction"])[["ip.src", "ip.dst"]].apply(
                                                                                        lambda x: list(numpy.unique(x)))
    ips = ips.to_frame().reset_index()
    ips.columns = ["ip.session_id", "meta.direction", "ip"]
    ips["ip.session_id"] = ips["ip.session_id"].astype(int)
    ips["meta.direction"] = ips["meta.direction"].astype(int)
    ip_traffic.drop(columns=["ip.src", "ip.dst"], inplace=True)

    ip_traffic = ip_traffic.apply(pd.to_numeric)
    tcp_traffic = tcp_traffic.apply(pd.to_numeric)
    tls_traffic = tls_traffic.apply(pd.to_numeric)
    ip_traffic['meta.direction'] = ip_traffic['meta.direction'].astype(int)
    tcp_traffic['meta.direction'] = tcp_traffic['meta.direction'].astype(int)
    tcp_traffic['tcp.src_port'] = tcp_traffic['tcp.src_port'].astype(int)
    tcp_traffic['tcp.dest_port'] = tcp_traffic['tcp.dest_port'].astype(int)
    tls_traffic['meta.direction'] = tls_traffic['meta.direction'].astype(int)

    ## deleting tcp and tls samples that have ip.session_id that was not present in ip_traffic (means that ip.session_id is wrongly assigned?)
    ids_tcp = tcp_traffic["ip.session_id"].unique().tolist()
    ids_ip = ip_traffic["ip.session_id"].unique().tolist()
    ids_tls = tls_traffic["ip.session_id"].unique().tolist()
    diff_tcp = set(ids_tcp) - set(ids_ip)
    diff_tls = set(ids_tls) - set(ids_ip)
    tcp_traffic = tcp_traffic[~tcp_traffic['ip.session_id'].isin(diff_tcp)]
    tls_traffic = tls_traffic[~tls_traffic['ip.session_id'].isin(diff_tls)]

    ip_traffic.set_index(["ip.session_id", "meta.direction"], inplace=True)
    tcp_traffic.set_index(["ip.session_id", "meta.direction"], inplace=True)

    ## Overall counters
    # total_traffic_nb = any_traffic['time'].count()  ## total number of any packets in csv
    # ip_total_nb = ip_traffic.groupby("ip.session_id")['time'].count().sum()  ## total number of ip packets in csv
    ip_pkts_per_flow = ip_traffic.groupby(["ip.session_id", "meta.direction"])['time'].count().reset_index().rename(
        columns={"time": "ip.pkts_per_flow"})  ## number of ip packets per session id

    ## Duration of flow: time between first and last received packet in one flow (i.e. in one direction per one session id)
    duration = ip_traffic.groupby(["ip.session_id", "meta.direction"])[
        ['ip.first_packet_time']].min().reset_index().merge(
        ip_traffic.groupby(["ip.session_id", "meta.direction"])[['ip.last_packet_time']].max().reset_index())
    duration['duration'] = duration['ip.last_packet_time'] - duration['ip.first_packet_time']

    features = ip_pkts_per_flow.merge(duration).drop(columns=['ip.first_packet_time', 'ip.last_packet_time'])
    duration = duration.iloc[0:0]
    ip_total_per_session = ip_pkts_per_flow.iloc[0:0]

    #####
    ip_header_len = ip_traffic.groupby(["ip.session_id", "meta.direction"])["ip.header_len"].sum().reset_index()
    features = features.merge(ip_header_len)

    ip_tot_len = ip_traffic.groupby(["ip.session_id", "meta.direction"])["ip.tot_len"].sum().reset_index().rename(
        columns={"ip.tot_len": "ip.bytes_tot_len"})  ### ?? TODO
    ip_tot_len["ip.payload_len"] = ip_tot_len["ip.bytes_tot_len"] - ip_header_len["ip.header_len"]
    ip_tot_len = ip_tot_len.drop(columns='ip.bytes_tot_len')
    features = features.merge(ip_tot_len)
    ip_header_len = ip_header_len.iloc[0:0]
    ip_tot_len = ip_tot_len.iloc[0:0]

    ip_avg_len = ip_traffic.groupby(["ip.session_id"])["ip.tot_len"].mean().reset_index().rename(
        columns={"ip.tot_len": "ip.avg_bytes_tot_len"})
    features = features.merge(ip_avg_len)
    ip_avg_len = ip_avg_len.iloc[0:0]

    # Packet Time
    ip_traffic['delta'] = (ip_traffic['time'] - ip_traffic['time'].shift()).fillna(0)
    ip_traffic['delta'] = ip_traffic['delta'] * 1000  # seconds to ms
    # df = ip_traffic.copy()
    # df = ip_traffic[['ip.session_id', 'meta.direction', 'delta']].copy()
    df = ip_traffic[['delta']].copy()
    #####

    print("Times between packets")
    time_between_pkts_sum = df.groupby(['ip.session_id', 'meta.direction'])['delta'].sum().reset_index().rename(
        columns={"delta": "time_between_pkts_sum"})
    time_between_pkts_avg = df.groupby(['ip.session_id', 'meta.direction'])['delta'].mean().reset_index().rename(
        columns={"delta": "time_between_pkts_avg"})
    time_between_pkts_max = df.groupby(['ip.session_id', 'meta.direction'])['delta'].max().reset_index().rename(
        columns={"delta": "time_between_pkts_max"})
    time_between_pkts_min = df.groupby(['ip.session_id', 'meta.direction'])['delta'].min().reset_index().rename(
        columns={"delta": "time_between_pkts_min"})
    time_between_pkts_std = df.groupby(['ip.session_id', 'meta.direction'])['delta'].std().reset_index().rename(
        columns={"delta": "time_between_pkts_std"})

    features = features.merge(time_between_pkts_sum)
    features = features.merge(time_between_pkts_avg)
    features = features.merge(time_between_pkts_max)
    features = features.merge(time_between_pkts_min)
    features = features.merge(time_between_pkts_std)

    time_between_pkts_sum = time_between_pkts_sum[0:0]
    time_between_pkts_avg = time_between_pkts_avg[0:0]
    time_between_pkts_max = time_between_pkts_max[0:0]
    time_between_pkts_min = time_between_pkts_min[0:0]
    time_between_pkts_std = time_between_pkts_std[0:0]

    print("SPTime Sequence")
    time = df.groupby(['ip.session_id', 'meta.direction'])['delta'].value_counts(bins=bins_time, sort=False).to_frame()
    df = df.iloc[0:0]
    time = time.rename(columns={'delta': 'county'}).reset_index()
    sptime = time.pivot_table(index=['ip.session_id', 'meta.direction'], columns='delta',
                              values='county')  # ,fill_value=0)
    sptime.columns = sptime.columns.astype(str)
    sptime = sptime.reset_index()

    features = features.merge(sptime)
    time = time.iloc[0:0]
    sptime = sptime.iloc[0:0]

    if not tcp_traffic.empty:
        print("TCP features")

        # TCP packets number per flow
        tcp_pkts_per_flow = tcp_traffic.groupby(["ip.session_id", "meta.direction"])[
            ['tcp.src_port']].count().reset_index().rename(
            columns={
                "tcp.src_port": "tcp_pkts_per_flow"})  ## number of tcp packets per flow, and per direction (0 = client->server)

        features = pd.merge(features, tcp_pkts_per_flow, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])

        features['pkts_rate'] = features['tcp_pkts_per_flow'] / features['duration']

        tcp_pkts_per_flow = tcp_pkts_per_flow.iloc[0:0]

        # # TCP bytes sum per flow
        tcp_bytes_per_flow = tcp_traffic.groupby(["ip.session_id", "meta.direction"])[
            ['tcp.payload_len']].sum().reset_index().rename(
            columns={
                "tcp.payload_len": "tcp_bytes_per_flow"})  ## sum of tcp bytes per flow per direction (0 = client->server)

        features = pd.merge(features, tcp_bytes_per_flow, how='outer', on=["ip.session_id", "meta.direction"])

        features['byte_rate'] = features['tcp_pkts_per_flow'] / features['duration']
        tcp_bytes_per_flow = tcp_bytes_per_flow.iloc[0:0]

        features = features.merge(tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
                                      'tcp.tcp_session_payload_up_len'].count().reset_index(), how='outer', on=["ip.session_id", "meta.direction"])
        features = features.merge(tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
                                      'tcp.tcp_session_payload_down_len'].count().reset_index(), how='outer', on=["ip.session_id", "meta.direction"])

        ## Sequence: Packet length and time sequences counted in bins, each bin stored as separate column
        # Packet length

        print("SPL Sequence")
        len = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])['tcp.payload_len'].value_counts(bins=bins_len,
                                                                                                       sort=False).to_frame()
        len = len.rename(columns={'tcp.payload_len': 'county'}).reset_index()

        # pivot_table to get columns out of segregated and divided packet lengths
        spl = len.pivot_table(index=['ip.session_id', 'meta.direction'], columns='tcp.payload_len',
                              values='county')  # ,fill_value=0)
        len = len.iloc[0:0]
        spl.columns = spl.columns.astype(str)
        spl = spl.reset_index()

        features = pd.merge(features, spl, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        spl = spl.iloc[0:0]

        print("Flags")
        # Flags: counts the number of turned on flags for each session and direction
        flag_list = ['tcp.fin', 'tcp.syn', 'tcp.rst', 'tcp.psh', 'tcp.ack', 'tcp.urg']
        tcp_flags_cnt_flow = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[flag_list].aggregate(
            lambda g: g.eq(
                1.0).sum()).reset_index()  # .drop(columns=['tcp.src_port', 'tcp.dest_port', 'tcp.payload_len','tcp.tcp_session_payload_up_len', 'tcp.tcp_session_payload_down_len'])

        features = pd.merge(features, tcp_flags_cnt_flow, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        tcp_flags_cnt_flow = tcp_flags_cnt_flow.iloc[0:0]

        ## Source and destination ports greater/less or equal to 1024 ( > ephemeral ports)
        # src ports
        print("Ports")
        # tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[['tcp.src_port']].apply(lambda x: len(x[x>3])/len(x) )
        sports = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[['tcp.src_port']].apply(
            lambda x: (x > 1024).sum()).reset_index().rename(columns={'tcp.src_port': 'sport_g'}).merge(
            tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[['tcp.src_port']].agg(
                lambda x: (x <= 1024).sum()).reset_index().rename(columns={'tcp.src_port': 'sport_le'})
        )

        features = pd.merge(features, sports, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])

        sports = sports.iloc[0:0]

        # dest port
        dports = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[['tcp.dest_port']].apply(
            lambda x: (x > 1024).sum()).reset_index().rename(columns={'tcp.dest_port': 'dport_g'}).merge(
            tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[['tcp.dest_port']].agg(
                lambda x: (x <= 1024).sum()).reset_index().rename(columns={'tcp.dest_port': 'dport_le'})
        )
        features = pd.merge(features, dports, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        dports = dports.iloc[0:0]

        print("Min/max pkts")

        mean_tcp_pkts = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.src_port'].mean().reset_index().rename(
            columns={"tcp.src_port": "mean_tcp_pkts"})
        std_tcp_pkts = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.src_port'].std().reset_index().rename(
            columns={"tcp.src_port": "std_tcp_pkts"})
        min_tcp_pkts = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.src_port'].min().reset_index().rename(
            columns={"tcp.src_port": "min_tcp_pkts"})
        max_tcp_pkts = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.src_port'].max().reset_index().rename(
            columns={"tcp.src_port": "max_tcp_pkts"})

        features = features.merge(mean_tcp_pkts, how='outer', on=["ip.session_id", "meta.direction"])
        features = features.merge(std_tcp_pkts, how='outer', on=["ip.session_id", "meta.direction"])
        features = features.merge(min_tcp_pkts, how='outer', on=["ip.session_id", "meta.direction"])
        features = features.merge(max_tcp_pkts, how='outer', on=["ip.session_id", "meta.direction"])
        mean_tcp_pkts = mean_tcp_pkts[0:0]
        std_tcp_pkts = std_tcp_pkts[0:0]
        min_tcp_pkts = min_tcp_pkts[0:0]
        max_tcp_pkts = max_tcp_pkts[0:0]

        print("Entropy pkts")
        entropy_tcp_pkts = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])['tcp.src_port'].apply(
            lambda x: entropy(
                x.value_counts(), base=2)).to_frame().reset_index().rename(columns={'tcp.src_port': 'entropy_tcp_pkts'})
        features = pd.merge(features, entropy_tcp_pkts, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])

        entropy_tcp_pkts = entropy_tcp_pkts.iloc[0:0]

        # Min, max, std and mean of packet length in each session+direction
        print("Min/max pkts")
        mean_tcp_len = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.payload_len'].mean().reset_index().rename(
            columns={"tcp.payload_len": "mean_tcp_len"})
        std_tcp_len = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.payload_len'].std().reset_index().rename(
            columns={"tcp.payload_len": "std_tcp_len"})
        min_tcp_len = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.payload_len'].min().reset_index().rename(
            columns={"tcp.payload_len": "min_tcp_len"})
        max_tcp_len = tcp_traffic.groupby(['ip.session_id', 'meta.direction'])[
            'tcp.payload_len'].max().reset_index().rename(
            columns={"tcp.payload_len": "max_tcp_len"})
        features = features.merge(mean_tcp_len, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        features = features.merge(std_tcp_len, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        features = features.merge(min_tcp_len, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        features = features.merge(max_tcp_len, how='outer', left_on=["ip.session_id", "meta.direction"],
                            right_on=["ip.session_id", "meta.direction"])
        mean_tcp_len = mean_tcp_len[0:0]
        std_tcp_len = std_tcp_len[0:0]
        min_tcp_len = min_tcp_len[0:0]
        max_tcp_len = max_tcp_len[0:0]

        print("Entropy len")

        #TODO: if MMT-probe will be able to provide any other attributes of TLS traffic they should be processed here
        if not tls_traffic.empty:
            print("TLS features")
            # TLS packets number per flow
            tls_pkts_per_flow = tls_traffic.groupby(["ip.session_id", "meta.direction"])[
                ['ssl.tls_version']].count().reset_index().rename(
                columns={"time": "tls_pkts_per_flow"})

            features = pd.merge(features, tls_pkts_per_flow, how='outer', on=["ip.session_id", "meta.direction"])
            tls_pkts_per_flow = tls_pkts_per_flow.iloc[0:0]

    #Features should have always same columns (as predefined), hence in case some features were not calculated due to
    # the lack of data (e.g. no TCP packets) the columns should be added anyway filled with 0 values
    features = features.reindex(features.columns.union(feature_names, sort=False), axis=1, fill_value=0)

    # ips = features['ip.session_id', 'meta.direction']
    # features.drop(columns=['ip.session_id', 'meta.direction'], inplace=True)
    # features.reset_index(inplace=True)
    # features.drop(columns=['delta'], inplace=True)

    print("Created {} feature samples".format(features.shape[0]))

    return ips, features
