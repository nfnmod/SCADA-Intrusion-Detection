import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import dataprocessing

test_captures_path = "C:\\Users\\User\\Desktop\\SCADA\\testcaptures"


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

        res = dataprocessing.process_data_v1(df, 4, dataprocessing.make_entry_v1)
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

        res = dataprocessing.process_data_v1(df, 4, dataprocessing.make_entry_v2)
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

        res = dataprocessing.process_data_v3(df, 4)
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

    def testv_2(self):
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


if __name__ == 'main':
    TestDataConversions()
