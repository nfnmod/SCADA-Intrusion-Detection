import datetime
from datetime import datetime
import unittest

import numpy as np
import pandas as pd

import data
import input


class TestInputPreparation(unittest.TestCase):

    def test_small_example(self):
        cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']

        datestrs = ["Mar 22, 2022 21:13:36.902262",
                    "Mar 22, 2022 21:13:36.903262",
                    "Mar 22, 2022 21:13:36.904262",
                    "Mar 22, 2022 21:13:36.905262",
                    "Mar 22, 2022 21:13:36.906262",
                    "Mar 22, 2022 21:13:36.907262",
                    "Mar 22, 2022 21:13:36.908262",
                    "Mar 22, 2022 21:13:36.909262",
                    "Mar 22, 2022 21:13:36.910262",
                    "Mar 22, 2022 21:13:36.911262"]

        times = [datetime.strptime(date_str, '%b %d, %Y %H:%M:%S.%f') for date_str in datestrs]

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

        stats_dict = data.get_plcs_values_statistics(pkt_df=df, n=10, to_df=False)
        sw_events, symbols, entities = input.define_events_in_sliding_windows(df, -1, -1, 4, stats_dict, True, False)
        expected_entities = {('1', 1,): 0, ('1', 2): 1, ('1', 3): 2, ('1', 4): 3, ('1', 9): 4}
        expected_entities_no_ids = set(expected_entities.keys())
        entities_set = set(entities)
        expected_symbols = {0: set(), 1: set(), 2: set(), 3: set(), 4: set()}
        for i in range(len(df)):
            curr_pkt = df.iloc[i]
            payload = curr_pkt['payload']
            for reg_num in payload.keys():
                if reg_num in ['1', '2', '3', '4', '9']:
                    expected_symbols[expected_entities[('1', int(reg_num))]].add(float(payload[reg_num]))
        compressed_expected_symbols = set()
        for entity_id in expected_symbols.keys():
            vals = expected_symbols[entity_id]
            reg_syms = [(entity_id, val,) for val in vals]
            compressed_expected_symbols = compressed_expected_symbols.union(set(reg_syms))
        symbols_no_ids = set(symbols.keys())
        assert symbols_no_ids.difference(
            compressed_expected_symbols) == set() and compressed_expected_symbols.difference(symbols_no_ids) == set()
        print('same symbols')
        assert entities_set.difference(expected_entities_no_ids) == set() and expected_entities_no_ids.difference(
                entities_set) == set()
        print('same entities')
        print(entities)
        print(symbols)
        print(sw_events)
