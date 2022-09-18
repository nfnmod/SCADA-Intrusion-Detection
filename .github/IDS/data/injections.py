"""
The code in this file injects time based anomalies into the test data. The anomalies change
in the length of the sequence of packets which is being infected, the change the anomaly causes to the
timings and the frequency of the anomalies in the data.
"""
from datetime import datetime

import numpy as np

import data

min_inter_arrival = 0
max_inter_arrival = 0


# works for data which has an inter-arrival time feature.
# not suitable for TIRP and automaton.
def inject_anomaly(x_test, injection_length, step_over, percentage):
    """
    :param x_test: the benign dataset of packets.
    :param injection_length: the number of subsequent infected packets in a series.
    :param step_over: the number of packets not to inject an anomaly into after injecting anomalies to injection_length packets
    :param percentage: determines how much should the timing be change, in the range of (-100, infinity) the closer to zero the
                        harder it should be to detect the anomaly.
    :return: test datasets with injected anomalies.
    """
    # first, flatten x_test for the injection.
    pkt_df = x_test[0]  # the first sequence of packets.
    n_features = len(pkt_df[0])  # each packet is an array. each element is a feature.
    series_len = len(pkt_df)
    # the sliding window for constructing the train and test sets slide 1 packet each time.
    for seq_num in range(1, len(x_test)):
        seq = x_test[seq_num]  # array of packets.
        new_packet = seq[-1]  # since we slide 1 packet a time.
        pkt_df.append(new_packet)

    # inject.
    i = 0
    y_labels = np.zeros((-1, len(x_test)))
    length = len(pkt_df)
    while i < length - injection_length + 1:
        # inject
        pkt_df.iloc[i: i + injection_length, 0] += (pkt_df.iloc[i: i + injection_length, 0] * (
                max_inter_arrival - min_inter_arrival) + min_inter_arrival) * (percentage / 100)
        # step over
        i += (step_over + injection_length)
        # save labels
        for packet_num in range(i, i + injection_length):
            if packet_num >= series_len:
                y_labels[packet_num - series_len] = 1
    # rescale the time.
    pkt_df['time'] = (pkt_df['time'] - min_inter_arrival) / (max_inter_arrival - min_inter_arrival)
    # now we reconstruct the data set but, now it has anomalies in it!
    y = []
    X_grouped = []

    # save the data as sequence of length series_len.
    i = 0

    while i < length - series_len:
        X_sequence = pkt_df.iloc[i:i + series_len, 0: n_features].to_numpy().reshape(series_len, n_features)
        y_sequence = pkt_df.iloc[i + series_len, 0: n_features].to_numpy().reshape(1, n_features)[0]

        X_grouped.append(X_sequence)
        y.append(y_sequence)
        i += 1

    X_grouped = np.array(X_grouped)
    y = np.array(y)
    x_test_anomalous = np.asarray(X_grouped).astype(np.float32)
    y_test_anomalous = np.asarray(y).astype(np.float32)
    return x_test_anomalous, y_test_anomalous, y_labels


def get_next_pkt_idx(pkt, test_data, j):
    """
    return the index of the next packet to/from the PLC which is used in the pkt (parameter).
    THIS WILL NOT WORK IF THE INDEX OF THE DATAFRAME ISN'T THE DEFAULT INDEX.
    :param pkt: the packet
    :param test_data: the data set
    :param j: the starting index (including j)
    :return: -1 if not found, it's index in test_df if found.
    """

    PLC = pkt['dst_ip']
    if pkt['src_port'] == data.plc_port:
        PLC = pkt['src_ip']

    pkts_indices = test_data.index(
        test_data.loc[(test_data['src_ip'] == PLC) | (test_data['dst_ip'] == PLC)]).tolist()
    if max(pkts_indices) < j:
        return -1
    else:
        for pkt_idx in sorted(pkts_indices):
            if pkt_idx >= j:
                return pkt_idx
        return -1


def get_prev_pkt_idx(pkt, test_data, j):
    """
    return the index of the previous packet to/from the PLC which is used in the pkt (parameter).
    THIS WILL NOT WORK IF THE INDEX OF THE DATAFRAME ISN'T THE DEFAULT INDEX.
    :param pkt: the packet
    :param test_data: the data set
    :param j: the starting index (going backwards)
    :return: -1 if not found, it's index in test_df if found.
    """

    PLC = pkt['dst_ip']
    if pkt['src_port'] == data.plc_port:
        PLC = pkt['src_ip']

    pkts_indices = test_data.index(
        test_data.loc[(test_data['src_ip'] == PLC) | (test_data['dst_ip'] == PLC)]).tolist()
    if min(pkts_indices) > j:
        return -1
    else:
        for pkt_idx in sorted(pkts_indices, reverse=True):
            if pkt_idx <= j:
                return pkt_idx
        return -1


def inject_to_raw_data(test_data, injection_length, step_over, percentage, epsilon):
    """
    inject anomalies via the time(date format) column.
    for TIRP: the output is fed to the input maker, then to KL, then to the classifier for testing.
    for the automaton: the output is fed into the data processing method and then to automaton for testing.
    :param epsilon: minimal time allowed between two subsequent packets.
    :param injection_length: the number of subsequent infected packets in a series.
    :param step_over: the number of packets not to inject an anomaly into after injecting anomalies to injection_length packets
    :param percentage: determines how much should the timing be change, in the range of (-100, infinity) the closer to zero the
                        harder it should be to detect the anomaly.
    :param test_data: this is the test part of the raw data. should only contain response packets since they are the only
    ones that carry information about a state transition or events.
    :return: the test data with injected time-anomalies and the labels of the test data packets.
    """
    i = 0
    length = len(test_data)
    labels = np.zeros(length)
    while i < length - injection_length + 1:
        # 1. get the "next" packet of the PLC (if exists) and use it as a limit of the change in the arrival time of the packet.
        # calculate new arrival time.
        if percentage > 0:
            for j in range(i + injection_length - 1, i - 1, -1):
                pkt = test_data.iloc[j]
                old_time = test_data.iloc[j, 0].total_seconds()
                next_pkt_idx = get_next_pkt_idx(pkt, test_data, j + 1)
                if next_pkt_idx != - 1:
                    limit = test_data.iloc[next_pkt_idx, 0].total_seconds() - epsilon
                    inter_arrival = (limit + epsilon) - old_time
                    new_time = min(old_time + inter_arrival * (1 + percentage), limit)
                else:
                    new_time = old_time
                test_data.iloc[j, 0] = datetime.fromtimestamp(new_time).strftime('%b %d, %Y %H:%M:%S.%f')
                labels[j] = 1
        else:
            for j in range(i, i + injection_length):
                pkt = test_data.iloc[j]
                old_time = test_data.iloc[j, 0].total_seconds()
                prev_pkt_idx = get_prev_pkt_idx(pkt, test_data, j - 1)
                if j != -1:
                    limit = test_data.iloc[prev_pkt_idx, 0].total_seconds() + epsilon
                    inter_arrival = (limit - epsilon) - old_time
                    new_time = max(old_time + inter_arrival * (1 + percentage), limit)
                else:
                    new_time = old_time
                test_data.iloc[j, 0] = datetime.fromtimestamp(new_time).strftime('%b %d, %Y %H:%M:%S.%f')
                labels[j] = 1
        i += (step_over + injection_length)
    return test_data, labels

