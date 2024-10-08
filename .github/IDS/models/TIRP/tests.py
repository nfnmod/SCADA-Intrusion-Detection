import datetime
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import data
import input
from models.TIRP import output
from output import get_number_of_instances

line = '2 143-245- <. 40 1.45 2 [11-13][41-54] 2 [27-33][41-54] 3 [12-14][41-48] 4 [1-18][27-34] 6 [1-7][24-33] 7 [10-12][40-49] 8 [8-10][11-17] 8 [8-10][29-34] 8 [34-36][51-56] 9 [1-5][11-15] 10 [1-6][25-29] 10 [32-37][60-65] 13 [1-10][15-18] 14 [22-31][49-58] 16 [10-13][42-43] 17 [1-3][10-13] 17 [1-3][31-36] 17 [13-16][31-36] 18 [1-5][21-24] 19 [4-11][14-17] 19 [4-11][32-34] 19 [33-35][49-51] 19 [37-40][49-51] 20 [1-6][26-27] 20 [1-6][30-34] 21 [1-7][10-11] 21 [1-7][23-26] 22 [8-12][27-32] 23 [1-8][24-27] 23 [10-12][24-27] 23 [29-31][54-60] 23 [64-66][91-94] 24 [8-11][25-32] 25 [8-13][36-38] 28 [1-3][20-29] 31 [14-16][43-57] 32 [14-18][37-48] 34 [16-19][36-46] 35 [9-12][35-49] 36 [26-31][53-65] 37 [11-19][27-33] 38 [47-49][71-72] 38 [51-54][71-72] 38 [65-66][71-72] 42 [1-9][20-22] 42 [1-9][30-34] 43 [3-16][35-41] 44 [1-9][22-29] 45 [10-16][22-27] 47 [32-36][63-64] 48 [1-5][24-29] 50 [20-21][37-45] 53 [12-20][44-54] 54 [18-26][47-57] 54 [37-41][47-57] 57 [1-9][32-41] 67 [31-42][46-55] 68 [29-39][55-66]'


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
        sw_events, symbols, entities = input.define_events_in_sliding_windows(df, -1, -1, 4, stats_dict, True, False,
                                                                              dict(), dict())
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
        assert entities_set.difference(expected_entities_no_ids) == set() and expected_entities_no_ids.difference(
            entities_set) == set()
        print('events')
        print(sw_events)
        print('symbols')
        print(symbols)
        print('entities')
        print(entities)

    def test_parse_raw_symbols(self):
        raw_syms = '143-245-'
        res = output.parse_raw_symbols(raw_syms)
        print('parsed symbols: {}'.format(res))

    def test_parse_raw_relations(self):
        raw_rels = '<.'
        res = output.parse_raw_relations(raw_rels)
        print('parsed relations: {}'.format(res))

    def test_parse_raw_instances(self):
        raw_instances = '2 [11-13][41-54] 2 [27-33][41-54] 3 [12-14][41-48] 4 [1-18][27-34] 6 [1-7][24-33] 7 [10-12][40-49] 8 [8-10][11-17] 8 [8-10][29-34] 8 [34-36][51-56] 9 [1-5][11-15] 10 [1-6][25-29] 10 [32-37][60-65] 13 [1-10][15-18] 14 [22-31][49-58] 16 [10-13][42-43] 17 [1-3][10-13] 17 [1-3][31-36] 17 [13-16][31-36] 18 [1-5][21-24] 19 [4-11][14-17] 19 [4-11][32-34] 19 [33-35][49-51] 19 [37-40][49-51] 20 [1-6][26-27] 20 [1-6][30-34] 21 [1-7][10-11] 21 [1-7][23-26] 22 [8-12][27-32] 23 [1-8][24-27] 23 [10-12][24-27] 23 [29-31][54-60] 23 [64-66][91-94] 24 [8-11][25-32] 25 [8-13][36-38] 28 [1-3][20-29] 31 [14-16][43-57] 32 [14-18][37-48] 34 [16-19][36-46] 35 [9-12][35-49] 36 [26-31][53-65] 37 [11-19][27-33] 38 [47-49][71-72] 38 [51-54][71-72] 38 [65-66][71-72] 42 [1-9][20-22] 42 [1-9][30-34] 43 [3-16][35-41] 44 [1-9][22-29] 45 [10-16][22-27] 47 [32-36][63-64] 48 [1-5][24-29] 50 [20-21][37-45] 53 [12-20][44-54] 54 [18-26][47-57] 54 [37-41][47-57] 57 [1-9][32-41] 67 [31-42][46-55] 68 [29-39][55-66]'
        res = output.parse_raw_instances(raw_instances, 2)
        print('parsed instances: {}'.format(res))

    def test_parse_line(self):
        res = output.parse_line('1 1- -. 1 1 1 [0-8] 1 [20-24] 1 [56-60] 1 [76-84] 1 [94-96] 1 [100-104] 1 [134-136] 1 [192-196] 1 [208-212] 1 [332-336] 1 [348-352] 1 [360-364] 1 [388-396]')
        print('tirp size is {}\n'.format(res.size))
        print('tirp symbols are {}\n'.format(res.symbols))
        print('tirp relations are {}\n'.format(res.relations))
        print('tirp instances are {}\n'.format(res.instances))
        print('tirp has {} supporting entites, {} instances'.format(res.supporting_entities, get_number_of_instances(res.instances)))

    def test_parse_line_second(self):
        res = output.parse_line(line)
        print('tirp size is {}\n'.format(res.size))
        print('tirp symbols are {}\n'.format(res.symbols))
        print('tirp relations are {}\n'.format(res.relations))
        print('tirp instances are {}\n'.format(res.instances))
        print('tirp has {} supporting entites, {} instances'.format(res.supporting_entities, get_number_of_instances(res.instances)))

    def test_parse_line_third(self):
        res = output.parse_line('3 3-2-3- m.m.s. 1 10 1 [8-20][20-24][24-28] 1 [32-40][40-44][44-48] 1 [58-64][64-68][68-72] 1 [184-192][192-196][196-200] 1 [204-212][212-216][216-224] 1 [216-224][224-228][228-232] 1 [264-272][272-276][276-284] 1 [276-284][284-288][288-304] 1 [364-380][380-384][384-392] 1 [384-392][392-397][397-400]')
        print('tirp size is {}\n'.format(res.size))
        print('tirp symbols are {}\n'.format(res.symbols))
        print('tirp relations are {}\n'.format(res.relations))
        print('tirp instances are {}\n'.format(res.instances))
        print('tirp has {} supporting entites, {} instances'.format(res.supporting_entities, get_number_of_instances(res.instances)))
