"""
The code in this file injects time based anomalies into the test data. The anomalies change
in the length of the sequence of packets which is being infected, the change the anomaly causes to the
timings and the frequency of the anomalies in the data.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import data


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
    mask = (test_data['src_ip'] == PLC) | (test_data['dst_ip'] == PLC)
    idx = test_data.index
    satisfying = idx[mask]
    pkts_indices = satisfying.tolist()
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

    idx = test_data.index
    mask = (test_data['src_ip'] == PLC) | (test_data['dst_ip'] == PLC)
    satisfying = idx[mask]
    pkts_indices = satisfying.tolist()
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
    :param epsilon: minimal time allowed between two subsequent packets. the unit is seconds.
    :param injection_length: the number of subsequent infected packets in a series.
    :param step_over: the number of packets not to inject an anomaly into after injecting anomalies to injection_length packets
    :param percentage: determines how much should the timing be change, in the range of (-100, infinity) the closer to zero the
                        harder it should be to detect the anomaly.
    :param test_data: this is the test part of the raw data. should only contain response packets since they are the only
    ones that carry information about a state transition or events.
    :return: the test data with injected time-anomalies and the labels of the test data packets.
    """
    test_data = data.reset_df_index(test_data)
    i = 0
    length = len(test_data)
    labels = np.zeros(length)
    cpy = test_data.copy()
    while i < length - injection_length + 1:
        # 1. get the "next" packet of the PLC (if exists) and use it as a limit of the change in the arrival time of the packet.
        # calculate new arrival time.
        if percentage > 0:
            for j in range(i + injection_length - 1, i - 1, -1):
                old_time = cpy.iloc[j, 0]
                next_pkt_idx = j + 1
                if next_pkt_idx >= len(test_data):
                    next_pkt_idx = -1
                if next_pkt_idx != - 1:
                    next_time = cpy.iloc[next_pkt_idx, 0]
                    # original inter-arrival time.
                    inter_arrival = (next_time - old_time).total_seconds()
                    # new inter-arrival time.
                    new_inter_arrival_time = inter_arrival * (1 - (percentage / 100))
                    # new arrival time.
                    new_time = test_data.iloc[next_pkt_idx, 0] - timedelta(seconds=new_inter_arrival_time)
                    if epsilon >= inter_arrival:
                        epsilon = inter_arrival / 2
                    max_limit = test_data.iloc[next_pkt_idx, 0] - timedelta(seconds=epsilon)
                    if new_time > max_limit:
                        new_time = max_limit
                    labels[j + 1] = 1
                    if i > 0:
                        labels[i] = 1
                    test_data.iloc[j, 0] = new_time
        else:
            for j in range(i, i + injection_length):
                old_time = cpy.iloc[j, 0]
                prev_pkt_idx = j - 1
                if j < 0:
                    j = -1
                next_pkt_idx = j + 1
                if next_pkt_idx >= len(test_data):
                    next_pkt_idx = -1
                if prev_pkt_idx != -1 and next_pkt_idx != -1:
                    nxt_time = cpy.iloc[next_pkt_idx, 0]
                    inter_arrival = (nxt_time - old_time).total_seconds()  # wrt to next one.
                    new_inter_arrival_time = inter_arrival * (1 - (percentage / 100))  # inc ia time
                    new_time = test_data.iloc[next_pkt_idx, 0] - timedelta(
                        seconds=new_inter_arrival_time)  # wrt to next one.
                    if epsilon >= inter_arrival:
                        epsilon = inter_arrival / 2
                    min_limit = test_data.iloc[prev_pkt_idx, 0] + timedelta(seconds=epsilon)
                    if new_time < min_limit:
                        new_time = min_limit
                    labels[j] = 1
                    test_data.iloc[j, 0] = new_time
        i += (step_over + injection_length)
    return test_data, labels


def inject_to_sub_group(test_data, injection_length, step_over, percentage, epsilon, plcs_to_effect):
    test_data_copy = test_data.copy()
    mask = (test_data_copy['src_ip'].isin(plcs_to_effect)) | (test_data_copy['dst_ip'].isin(plcs_to_effect))
    data_to_change = test_data_copy.loc[mask]
    data_not_to_change = test_data_copy.loc[~mask]

    data_to_change, data_to_change_labels = inject_to_raw_data(data_to_change, injection_length, step_over, percentage,
                                                               epsilon)

    df = pd.concat([data_not_to_change, data_to_change], ignore_index=True).sort_values(by=['time'])
    df = data.reset_df_index(df)

    labels = []

    for i in range(len(test_data_copy)):
        if test_data_copy.loc[i, 'time'] != df.loc[i, 'time']:
            labels.append(1)
        else:
            labels.append(0)

    return df, labels
