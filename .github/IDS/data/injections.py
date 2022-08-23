"""
The code in this file injects time based anomalies into the test data. The anomalies change
in the length of the sequence of packets which is being infected, the change the anomaly causes to the
timings and the frequency of the anomalies in the data.
"""
import numpy as np

min_inter_arrival = 0
max_inter_arrival = 0


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
