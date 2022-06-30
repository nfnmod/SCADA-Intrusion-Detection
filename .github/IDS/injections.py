"""
The code in this file injects time based anomalies into the data. The anomalies change
in the length of the sequence of packets which is being infected, the change the anomaly causes to the
timings and the frequency of the anomalies in the data.
"""
import pandas as pd


def inject_anomaly(pkt_df, injection_length, step_over, percentage):
    """
    :param pkt_df: the benign dataset of packets.
    :param injection_length: the number of subsequent infected packets in a series.
    :param step_over: the number of packets not to inject an anomaly into after injecting anomalies to injection_length packets
    :param percentage: determines how much should the timing be change, in the range of (-100, infinity) the closer to zero the
                        harder it should be to detect the anomaly.
    :return: a dataset with injected anomalies.
    """
    pkt_df['anomaly'] = 0
    i = 0
    length = len(pkt_df)
    while i < length - injection_length + 1:
        # inject
        pkt_df.iloc[i: i + injection_length, 0] += pkt_df.iloc[i: i + injection_length, 0] * (percentage / 100)
        pkt_df.iloc[i: i + injection_length, len(pkt_df.columns) - 1] = 1
        # step over
        i += (step_over + injection_length)
    return pkt_df


if __name__ == '__main__':
    pkt_df = pd.DataFrame(data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                          columns=['time'])
    print(inject_anomaly(pkt_df, 3, 3, 100))
