import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import data

test_captures_path = "C:\\Users\\User\\Desktop\\SCADA\\testcaptures"


def est_find_frequent_basic_transitions():
    return None


def est_find_frequent_extended_transitions():
    return None


class TestDataConversions(unittest.TestCase):

    # need to cancel the scaling to run this correctly
    def est1(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.933460",
                    "Mar 22, 2022 21:13:37.127180",
                    "Mar 22, 2022 21:13:37.153783",
                    "Mar 22, 2022 21:13:37.308402",
                    "Mar 22, 2022 21:13:37.339604",
                    "Mar 22, 2022 21:13:37.714403",
                    "Mar 22, 2022 21:13:37.939062",
                    "Mar 22, 2022 21:13:37.965069",
                    "Mar 22, 2022 21:13:38.750953"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 8), np.repeat(54312, 2)))

        func_codes = np.repeat(3, 10)

        registers_freqs = {'1': 7, '2': 7, '3': 7, '4': 7, '9': 8}

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {'1': 45, '2': 43, '3': 43, '4': 34, '9': 444},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        res = data.dataprocessing.process_data_v1(df, 4, data.dataprocessing.make_entry_v1)
        col9 = [0, 11, 0, 112, 1, 444, 0, 0, 0]
        col4 = [42, 65, 65, 876, 10, 34, 34, 34, 34]
        col3 = [11, 222, 222, 765, 0, 43, 8, 8, 8]
        col2 = [654, 34, 34, 543, 0, 43, 10, 10, 10]
        check = np.alltrue([col9 == res['9'], col4 == res['4'], col3 == res['3'], col2 == res['2']])
        print(res)
        print(check)

    # need to cancel the scaling to run this correctly

    def est1_v2(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.933460",
                    "Mar 22, 2022 21:13:37.127180",
                    "Mar 22, 2022 21:13:37.153783",
                    "Mar 22, 2022 21:13:37.308402",
                    "Mar 22, 2022 21:13:37.339604",
                    "Mar 22, 2022 21:13:37.714403",
                    "Mar 22, 2022 21:13:37.939062",
                    "Mar 22, 2022 21:13:37.965069",
                    "Mar 22, 2022 21:13:38.750953"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 8), np.repeat(54312, 2)))

        func_codes = np.repeat(3, 10)

        registers_freqs = {'1': 7, '2': 7, '3': 7, '4': 7, '9': 8}

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {'1': 45, '2': 43, '3': 43, '4': 34, '9': 444},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        res = data.dataprocessing.process_data_v1(df, 4, data.dataprocessing.make_entry_v2)
        col9 = [0, 11, 11, 112, 111, 443, 444, 71, 71]
        print(col9 == res['9'])

    # turn off scaling and binning for this to work
    def est3(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.933460",
                    "Mar 22, 2022 21:13:37.127180",
                    "Mar 22, 2022 21:13:37.153783",
                    "Mar 22, 2022 21:13:37.308402",
                    "Mar 22, 2022 21:13:37.339604",
                    "Mar 22, 2022 21:13:37.714403",
                    "Mar 22, 2022 21:13:37.939062",
                    "Mar 22, 2022 21:13:37.965069",
                    "Mar 22, 2022 21:13:38.750953"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 6), np.repeat(54312, 1), np.repeat(502, 1), np.repeat(54312, 2)))

        func_codes = np.repeat(3, 10)

        registers_freqs = {'1': 7, '2': 7, '3': 7, '4': 7, '9': 8}

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 1},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        res = data.dataprocessing.process_data_v3(df, 4)
        col9 = [0, 11, 0, 112, 1, 1]
        col4 = [42, 65, 65, 876, 10, 34]
        col3 = [11, 222, 222, 765, 0, 8]
        col2 = [654, 34, 34, 543, 0, 10]
        col_msgs_in_state = [1, 1, 1, 1, 2, 3]
        col_time_in_state = [times[2] - times[1], times[3] - times[2], times[4] - times[3], times[5] - times[4],
                             times[7] - times[5], times[9] - times[7]]
        col_time_in_state = [t.total_seconds() for t in col_time_in_state]
        check = np.alltrue([col9 == res['9'], col4 == res['4'], col3 == res['3'], col2 == res['2'],
                            col_msgs_in_state == res['msgs_in_state'],
                            col_time_in_state == res['time_in_state']])
        print(check)

    def estv_2(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.933460",
                    "Mar 22, 2022 21:13:37.127180",
                    "Mar 22, 2022 21:13:37.153783",
                    "Mar 22, 2022 21:13:37.308402",
                    "Mar 22, 2022 21:13:37.339604",
                    "Mar 22, 2022 21:13:37.714403",
                    "Mar 22, 2022 21:13:37.939062",
                    "Mar 22, 2022 21:13:37.965069",
                    "Mar 22, 2022 21:13:38.750953"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 8), np.repeat(54312, 2)))

        func_codes = np.repeat(3, 10)

        registers_freqs = {'1': 7, '2': 7, '3': 7, '4': 7, '9': 8}

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {'1': 45, '2': 43, '3': 43, '4': 34, '9': 444},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        # res = dataprocessing.process_data_v2(df, 4)
        col9 = [0, 11, 0, 112, 1, 444, 0, 0, 0]
        col2 = [654, 34, 34, 543, 0, 43, 10, 10, 10]
        col_time_9 = [times[2] - times[1], times[3] - times[2], times[4] - times[3], times[5] - times[4],
                      times[6] - times[5], times[7] - times[6], times[0] - times[0],
                      times[8] - times[7], times[9] - times[7]]
        col_time_2 = [times[2] - times[1], times[0] - times[0], times[4] - times[2], times[5] - times[4],
                      times[6] - times[5], times[7] - times[6], times[0] - times[0], times[8] - times[7],
                      times[9] - times[7]]
        col_time_9 = [t.total_seconds() for t in col_time_9]
        col_time_2 = [t.total_seconds() for t in col_time_2]
        # check = np.alltrue(
        # [col9 == res['9'], col2 == res['2'], col_time_2 == res['time_2'], col_time_9 == res['time_9']])
        # print(check)

    def est_embedding(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.933460",
                    "Mar 22, 2022 21:13:37.127180",
                    "Mar 22, 2022 21:13:37.153783",
                    "Mar 22, 2022 21:13:37.308402",
                    "Mar 22, 2022 21:13:37.339604",
                    "Mar 22, 2022 21:13:37.714403",
                    "Mar 22, 2022 21:13:37.939062",
                    "Mar 22, 2022 21:13:37.965069",
                    "Mar 22, 2022 21:13:38.750953"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 8), np.repeat(54312, 2)))

        func_codes = np.repeat(3, 10)

        registers_freqs = {'1': 7, '2': 7, '3': 7, '4': 7, '9': 8}

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {'1': 45, '2': 43, '3': 43, '4': 34, '9': 444},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        processed_df = data.dataprocessing.embedding_v1(df, 3,
                                                        regs_times_maker=data.dataprocessing.embed_v1_with_deltas_regs_times,
                                                        scale=False)
        print(processed_df)

    def est_injection(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.902262"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]
        m = [100] * 10
        for i in range(len(times)):
            times[i] += timedelta(seconds=m[i] * i)

        original = []
        for i in range(len(times) - 1):
            p = times[i]
            n = times[i + 1]
            ia = (n - p).total_seconds()
            original.append(ia)

        dst_ips = np.repeat('0', 10)
        src_ips = np.repeat('1', 10)

        dst_ports = np.repeat(80, 10)
        src_ports = np.concatenate((np.repeat(502, 8), np.repeat(502, 2)))

        func_codes = np.repeat(3, 10)

        payloads = [{'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {'1': 42, '2': 654, '3': 11, '4': 42, '9': 0},
                    {'1': 76, '2': 34, '3': 222, '4': 65, '9': 11},
                    {'7': 45, '6': 10, '5': 8, '13': 34, '9': 0},
                    {'1': 4543, '2': 543, '3': 765, '4': 876, '9': 112},
                    {'1': 54, '2': 0, '3': 0, '4': 10, '9': 1},
                    {'1': 45, '2': 43, '3': 43, '4': 34, '9': 444},
                    {'1': 45, '2': 10, '3': 8, '4': 34, '9': 0},
                    {},
                    {}]

        df = pd.DataFrame(columns=cols)
        df['time'] = times
        df['dst_ip'] = dst_ips
        df['src_ip'] = src_ips
        df['dst_port'] = dst_ports
        df['src_port'] = src_ports
        df['func_code'] = func_codes
        df['payload'] = payloads

        def check(expected_labels, expected_ia):
            for i in range(len(expected_labels)):
                assert expected_labels[i] == labels[i]
            for i in range(len(expected_ia)):
                assert expected_ia[i] == inter_arrivals[i]

        def make_ia(df):
            inter_arrivals = []
            for i in range(len(df) - 1):
                c = df.iloc[i]
                n = df.iloc[i + 1]
                ia = (n['time'] - c['time']).total_seconds()
                inter_arrivals.append(ia)
            return inter_arrivals

        cpy = df.copy()
        print(cpy)
        injected, labels = data.injections.inject_to_raw_data(df, 3, 2, 50, 0.00001)
        inter_arrivals = []
        inter_arrivals = make_ia(df)
        expected_ia = [100, 100, 50, 100, 150, 100, 100, 50, 100]
        expected_labels = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        check(expected_labels, expected_ia)
        injected, labels = data.injections.inject_to_raw_data(cpy, 3, 2, -50, 0.00001)
        inter_arrivals = make_ia(cpy)
        expected_ia = [50, 100, 150, 100, 50, 100, 100, 150, 100]
        expected_labels = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        check(expected_labels, expected_ia)

    def test_find_frequent_states(self):
        df = pd.DataFrame(columns=['time', 'time_in_state', '1'])
        times = [0] * 8
        times_in_state = [13, 11, 12, 10, 8, 1, 2, 7]
        reg_values = [8, 9, 10, 8, 10, 11, 1, 8]

        df['time'] = times
        df['time_in_state'] = times_in_state
        df['1'] = reg_values

        states_appearances, frequent_states, packets = data.PLCDependeciesAlgorithm.get_base_states(
            df, ['1'], 25)
        expected_states_appearances = {(('1', 8,),): [0, 3, 7], (('1', 9,),): [1], (('1', 10,),): [2, 4],
                                       (('1', 1,),): [6],
                                       (('1', 11,),): [5]}
        for i in range(len(packets)):
            timestamp = packets.loc[i, 'timestamp']
            expected = 0
            for j in range(i):
                expected += packets.loc[j, 'time_in_state']
            assert expected == timestamp

        for k in expected_states_appearances.keys():
            assert expected_states_appearances[k] == states_appearances[k]

        expected_frequent_states = [(('1', 8,),), (('1', 10,),)]

        assert len(expected_frequent_states) == len(frequent_states)
        for frequent_state in expected_frequent_states:
            assert frequent_state in frequent_states

    def test_find_frequent_base_transitions_no_frequent_transitions(self):
        df = pd.DataFrame(columns=['time', 'time_in_state', '1'])
        times = [0] * 12
        times_in_state = [2, 8, 1, 1, 2, 10, 5, 4, 1, 7, 17, 15]
        reg_values = [0, 1, 2, 8, 11, 2, 0, 1, 8, 11, 0, 1]
        window = 10
        length = 12

        df['time'] = times
        df['time_in_state'] = times_in_state
        df['1'] = reg_values
        states_appearances, frequent_states, packets = data.PLCDependeciesAlgorithm.get_base_states(
            df, ['1'], 25)
        frequent_transitions, transitions_times, transitions_indices = data.PLCDependeciesAlgorithm.base_transitions(
            frequent_states, states_appearances, df, window, length, 25)

        assert len(frequent_transitions) == 0
        assert len(transitions_times.keys()) == 1
        assert len(transitions_indices.keys()) == 1
        assert list(transitions_times.keys())[0] == ((('1', 0),), (('1', 1),),) == list(transitions_indices.keys())[0]

    def test_find_frequent_base_transitions_found_frequent_transitions(self):
        df = pd.DataFrame(columns=['time', 'time_in_state', '1'])
        times = [0] * 12
        times_in_state = [2, 8, 1, 1, 2, 10, 5, 4, 1, 7, 7, 15]
        reg_values = [0, 1, 2, 8, 11, 2, 0, 1, 8, 11, 0, 1]
        window = 10
        length = 12

        df['time'] = times
        df['time_in_state'] = times_in_state
        df['1'] = reg_values
        states_appearances, frequent_states, packets = data.PLCDependeciesAlgorithm.get_base_states(
            df, ['1'], 25)
        frequent_transitions, transitions_times, transitions_indices = data.PLCDependeciesAlgorithm.base_transitions(
            frequent_states, states_appearances, df, window, length, 25)

        k = ((('1', 0),), (('1', 1),),)

        assert len(frequent_transitions) == 1
        assert len(transitions_times.keys()) == 1
        assert len(transitions_indices.keys()) == 1
        assert list(transitions_times.keys())[0] == ((('1', 0),), (('1', 1),),) == list(transitions_indices.keys())[0]
        assert len(transitions_times[k]) == 3 and 2.0 in transitions_times[k] and 5.0 in transitions_times[k] and 7.0 in \
               transitions_times[k]
        assert len(transitions_indices[k]) == 3 and (0, 1) in transitions_indices[k] and (6, 7) in transitions_indices[
            k] and (10, 11) in transitions_indices[k]

    def test_find_long_transitions(self):
        df = pd.DataFrame(columns=['time', 'time_in_state', '1'])
        times = [0] * 20
        times_in_state = [1, 9, 2, 3, 15, 5, 4, 8, 8, 7, 2, 6, 8, 3, 3, 1, 1, 2, 4, 20]
        reg_values = [8, 0, 1, 2, 8, 2, 3, 4, 7, 2, 3, 4, 0, 1, 0, 2, 3, 4, 7, 19]
        window = 10

        df['time'] = times
        df['time_in_state'] = times_in_state
        df['1'] = reg_values
        flat_transitions, prev_times, prev_indices, longest, time_stamp = data.PLCDependeciesAlgorithm.find_frequent_transitions_sequences(
            df, window, 10)

        for f_t in flat_transitions:
            print(f_t)


if __name__ == 'main':
    TestDataConversions()
