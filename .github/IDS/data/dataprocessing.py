import csv
import itertools
import json
import os
import pickle
import statistics
from datetime import datetime

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from sklearn.model_selection import train_test_split
from pathlib import Path

import models
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

plc_port = 502
captures_path = 'C:\\Users\\User\\Desktop\\SCADA\\modbuscaptures'
datasets_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\datasets'
modeles_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\models'
automaton_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\DFA'
plots_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\plots\\regular\\singleplc'
excel_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\excel'
plc = '132.72.249.110'
to_bin = ['30', '120', '15']
most_used = ['30', '75', '120', '195', '15']

# TODO: configuration file for algorithms parameters used for tests and training.
# TODO: add log files for running times.
# TODO: excel columns for algorithms parameters, metrics, names.
# TODO: write code for algorithms performance comparisons.
# TODO: write code for injections of anomalies.
# TODO: determine parameters values for the KL grid- calculate statistics in the data to determine them better.
# TODO: fix warnings.
# TODO: test input.py , output.py.
# TODO: find out min and max inter arrival times for injection of anomalies.

# TODO (last, not urgent):  retrain all LSTMs and record their scores and training time, manual grid search and KFold-CV.
# ---------------------------------------------------------------------------------------------------------------------------#
# helper function used to perform min-max scaling on a single column
def scale_col(df, name):
    scaler = MinMaxScaler()
    np_col = df[name].to_numpy().reshape(-1, 1)
    scaler.fit(np_col)
    return scaler.transform(np_col)


# gets the average registers values of a PLC
def get_avg_vals(pkt_df, plc_ip, registers):
    # calculate the average value of a register based on number of time read and sum of values read
    def get_avg(reg_pair):
        if reg_pair[1] == 0:
            return np.nan
        else:
            return reg_pair[0] / reg_pair[1]

    # read 0 times
    vals_dict = {reg: [0, 0] for reg in registers}
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == plc_ip) | (pkt_df['src_ip'] == plc_ip)]

    for i in range(len(plc_pkts)):
        pkt = plc_pkts.iloc[i]
        src_port = pkt['src_port']
        # a value has been reported by the plc
        if src_port == plc_port:
            payload = pkt['payload']
            for reg in registers:
                # for each register that we consider and has changed
                if reg in payload.keys():
                    reg_pair = vals_dict[reg]
                    # read 1 more time and the sum is increased
                    vals_dict[reg] = [reg_pair[0] + int(payload[reg]), reg_pair[1] + 1]
    # average the values
    vals_avg_dict = {reg: get_avg(vals_dict[reg]) for reg in registers}

    return vals_avg_dict


# update the frequency dictionary
def update_freqs(current_freqs, payload):
    registers = list(payload.keys())

    for reg_num in registers:
        current_freqs[reg_num] = current_freqs.get(reg_num, 0) + 1
    return current_freqs


# return the n most used registers by every PLC
def get_frequent_registers(pkt_df, n):
    registers_freq = {}

    for i in range(len(pkt_df)):
        row = pkt_df.iloc[i]
        src_port = row['src_port']
        # this is a response packet
        if src_port == 502:
            plc_ip = row['src_ip']
            registers_freq[plc_ip] = update_freqs(registers_freq.get(plc_ip, {}), row['payload'])

    sorted_freqs = {k: sorted(v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True) for k, v in
                    registers_freq.items()}
    frequent_regs = {k: list(map(lambda t: t[0], v))[0: min(n, len(v))] for k, v in sorted_freqs.items()}
    return frequent_regs


# return the most commonly used PLC. f represents the choosing criteria.
def get_frequent_plc(pkt_df):
    def calc_ip(plc_freq_responses_top, plc_freq_queries_top, plc_freq_readvalues_top, consider_all=False):
        # get the ips
        ips_rv = [ip for (ip, val) in plc_freq_readvalues_top]
        ips_r = [ip for (ip, val) in plc_freq_responses_top]
        ips_q = [ip for (ip, val) in plc_freq_queries_top]
        # find ip which are in more than 1 group
        q_r_v = list(set(ips_q) & set(ips_r) & set(ips_rv))
        q_r = list(set(ips_q) & set(ips_r))

        if q_r_v and consider_all:
            IP = None
            min_score = np.nan
            for plc in q_r_v:
                score = 0.6 * ips_r.index(plc) + 0.2 * ips_q.index(plc) + 0.2 * ips_rv.index(plc)
                if IP is None or score < min_score:
                    IP = plc
                    min_score = score

            return IP
        # top responses and top queries , compare the plcs using a score
        elif q_r:
            IP = None
            min_score = np.nan
            for plc in q_r:
                score = 0.6 * ips_r.index(plc) + 0.4 * ips_q.index(plc)
                if IP is None or score < min_score:
                    IP = plc
                    min_score = score

            return IP
        else:
            IP = ips_r[0]

            return IP

    plc_freq_readvalues = {}
    plc_freq_responses = {}
    plc_freq_queries = {}

    for i in range(len(pkt_df)):
        row = pkt_df.iloc[i]
        src_port = row['src_port']
        if src_port == 502:
            plc_ip = row['src_ip']
            payload = row['payload']
            score = len(payload.keys())
            # update entries
            plc_freq_readvalues[plc_ip] = plc_freq_readvalues.get(plc_ip, 0) + score
            plc_freq_responses[plc_ip] = plc_freq_responses.get(plc_ip, 0) + 1
        else:
            # one more query
            plc_ip = row['dst_ip']
            plc_freq_queries[plc_ip] = plc_freq_queries.get(plc_ip, 0) + 1

    sorted_plc_freq_readvalues = sorted(plc_freq_readvalues.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    sorted_plc_freq_responses = sorted(plc_freq_responses.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    sorted_plc_freq_queries = sorted(plc_freq_queries.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    plc_freq_readvalues_top = sorted_plc_freq_readvalues[0:15]
    plc_freq_responses_top = sorted_plc_freq_responses[0:15]
    plc_freq_queries_top = sorted_plc_freq_queries[0:15]

    ip = calc_ip(plc_freq_responses_top, plc_freq_queries_top, plc_freq_readvalues_top)

    return ip


# return the PLC which had the most state switches considering the n most used registers
def get_plc(pkt_df, n):
    # map IP to frequent register
    regs_dict = get_frequent_registers(pkt_df, n)
    # get only PLC where there are at least n used registers
    filtered_dict = {k: v for k, v in regs_dict.items() if len(v) == n}
    # ips of those plcs
    ips = filtered_dict.keys()
    # the last state knows for each plc, map register to last known value
    last_states = {ip: {reg: None for reg in filtered_dict[ip]} for ip in ips}
    # number of state switches for each plc
    switches = {ip: 0 for ip in ips}
    for ip in ips:
        # packets with that plc
        plc_df = pkt_df.loc[(pkt_df['dst_ip'] == ip) | (pkt_df['src_ip'] == ip)]
        # registers we look at
        regs = regs_dict[ip]
        for i in range(0, len(plc_df)):
            curr_pkt = plc_df.iloc[i]
            src_port = curr_pkt['src_port']
            # a response packet
            if src_port == plc_port:
                payload = curr_pkt['payload']
                last_state = last_states[ip]
                # check if this is the first time we see some register
                change_none = {reg: (last_state[reg] is None and payload.get(reg, None) is not None) for reg in regs}
                # check if this isn't the first time we see a register and we see a different value
                change_not_none = {
                    reg: (last_state[reg] is not None and payload.get(reg, last_state[reg]) != last_state[reg]) for reg
                    in regs}
                # was there any change?
                change = {reg: change_none[reg] or change_not_none[reg] for reg in regs}
                if np.any(list(change.values())):
                    switches[ip] = switches[ip] + 1
                # build the new last know state
                new_last_state = {reg: None for reg in regs}
                for reg in regs:
                    if change[reg]:
                        new_last_state[reg] = payload[reg]
                    else:
                        new_last_state[reg] = last_state[reg]
                last_states[ip] = new_last_state
    sorted_switches = sorted(switches.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(sorted_switches)


# works for a single PLC.
def get_frequent_registers_values(pkts, registers):
    # map a register to a dictionary. the dict maps values to frequencies
    regs_vals = {r: dict() for r in registers}
    for i in range(len(pkts)):
        curr = pkts[i]
        src_port = curr['src_port']
        # response packet
        if src_port == plc_port:
            payload = curr['payload']
            changed = payload.keys()
            for r in registers:
                # value of register
                val = payload[r]
                if r in changed:
                    # the dict
                    reg_values = regs_vals[r]
                    # increase frequency of value
                    reg_values[val] = reg_values.get(val, 0) + 1
    for r in registers:
        r_dict = regs_vals[r]
        # sort the values by frequencies, get the most frequent one , get only the value.
        regs_vals[r] = sorted(r_dict, key=lambda kv: (kv[1], kv[0]), reverse=True)[0][1]
    return regs_vals


# using RAW DATA
def get_plcs_values_statistics(pkt_df, n, to_df=True):
    # map IP to frequent registers
    regs_dict = get_frequent_registers(pkt_df, n)
    # get only PLC where there are at least n used registers
    if to_df:
        filtered_dict = {k: v for k, v in regs_dict.items() if len(v) == n}
    else:
        filtered_dict = {k: v for k, v in regs_dict.items()}
    stats_dict = {ip: None for ip in regs_dict.keys()}
    # ips of those plcs
    ips = filtered_dict.keys()
    # the last state knows for each plc, map register to last known value
    regs_vals = {ip: {reg: None for reg in filtered_dict[ip]} for ip in ips}
    cols = np.concatenate((['ip'], ["register " + str(i) + " #values" for i in range(1, n + 1)],
                           ["register " + str(i) + " stdev" for i in range(1, n + 1)]))
    stats_df = pd.DataFrame(columns=cols)
    for ip in ips:
        # packets with that plc
        plc_df = pkt_df.loc[(pkt_df['dst_ip'] == ip) | (pkt_df['src_ip'] == ip)]
        # registers we look at
        regs = regs_dict[ip]
        for i in range(0, len(plc_df)):
            curr_pkt = plc_df.iloc[i]
            src_port = curr_pkt['src_port']
            # a response packet
            if src_port == plc_port:
                payload = curr_pkt['payload']
                # first time we see the register
                change_none = {reg: (regs_vals[ip][reg] is None and payload.get(reg, None) is not None) for reg in regs}
                # we see a new value
                change_not_none = {
                    reg: (regs_vals[ip][reg] is not None and payload.get(reg, None) not in regs_vals[ip][
                        reg] and payload.get(reg, None) is not None) for reg
                    in regs}
                # one of the options
                change = {reg: change_none[reg] or change_not_none[reg] for reg in regs}
                if np.any(list(change.values())):
                    # add values to the lists
                    for reg in regs:
                        if change[reg]:
                            # no previous knows values
                            if regs_vals[ip][reg] is None:
                                regs_vals[ip][reg] = [payload[reg]]
                            # new value, so add it to the list
                            else:
                                regs_vals[ip][reg].append(payload[reg])
        vals = regs_vals[ip]
        counter = 1
        entry = {'ip': ip}
        stats_entry = {}
        for reg, vs in vals.items():
            values = vals[reg]
            values = [np.float64(v) for v in values]
            std_v = np.std(values)
            num_vals = len(values)
            entry["register " + str(counter) + " #values"] = num_vals
            entry["register " + str(counter) + " stdev"] = std_v
            stats_entry[reg] = [num_vals, std_v]
            counter += 1
        if to_df:
            temp_df = pd.DataFrame.from_dict(columns=stats_df.columns,
                                             data={'0': [entry[col] for col in stats_df.columns]}, orient='index')
            stats_df = pd.concat([stats_df, temp_df], ignore_index=True)
        else:
            stats_dict[ip] = stats_entry
    if to_df:
        return stats_df
    else:
        return stats_dict


def get_most_frequently_switching_registers(pkt_df, plc_ip, n_reg):
    # relevant response packets to the given plc
    plc_df = pkt_df.loc[pkt_df["src_ip"] == plc_ip]
    registers_switches = dict()
    register_last_values = dict()
    # iterate over dataframe, calculate registers switches
    for i in range(0, len(plc_df)):
        response = plc_df.iloc[i]
        payload = response['payload']
        for register in payload.keys():
            # last value known for this register
            last_val = register_last_values.get(register, np.NAN)
            # first time we see it in a payload of a packet
            if last_val == np.NAN:
                register_last_values[register] = payload[register]
                registers_switches[register] = 1
            else:
                # we have recorded a value for it before
                curr_val = payload[register]
                if curr_val != last_val:
                    register_last_values[register] = curr_val
                    registers_switches[register] += 1
    # sort register in descending order by the number of value changes
    sorted_switches = sorted(registers_switches, key=lambda kv: (kv[1], kv[0]), reverse=True)
    # get only registers which have received more than 1 value
    sorted_switches = filter(lambda kv: kv[1] > 1, sorted_switches)
    return sorted_switches


def view_times(df):
    times = []
    for i in range(1, len(df)):
        c = df.iloc[i]
        p = df.iloc[i - 1]
        t = (c['time'] - p['time']).total_seconds()
        times.append(t)
    t_df = pd.DataFrame(data=times, columns=['time'])
    with open(excel_path + "\\times.xlsx", 'w'):
        t_df.to_excel(excel_path + "\\times.xlsx")


# make a df with: number of switches , number of packets staying in the same state , average switch frequency of the PLCs' register values.
def get_switch_freq(export_to_excel=True, sheet_name="switch_freq_sheet"):
    pkt_df = load(datasets_path, "modbus")
    IP = plc

    # filter out the packets that don't involve the most frequently used PLC
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == IP) | (pkt_df['src_ip'] == IP)]

    registers = to_bin
    last_vals = {r: None for r in registers}
    stayed_same = {r: 0 for r in registers}
    switches = {r: 0 for r in registers}
    cols = np.concatenate((["switches " + r for r in registers], ["stayed same " + r for r in registers],
                           ["average switch freq" + r for r in registers]))
    switch_df = pd.DataFrame(columns=cols)

    for i in range(len(plc_pkts)):
        pkt = plc_pkts.iloc[i]
        src_port = pkt['src_port']

        # it's a response packet
        if src_port == plc_port:
            changed_regs = pkt['payload'].keys()
            payload = pkt['payload']
            for register in registers:
                # check if updated this register in the current packet
                if register in changed_regs:
                    # first time seeing the register
                    if last_vals[register] is None or (
                            last_vals[register] is not None and last_vals[register] != payload[register]):
                        last_vals[register] = payload[register]
                        switches[register] = switches[register] + 1
                    else:
                        # same value as the last one known
                        stayed_same[register] = stayed_same[register] + 1
                else:
                    # this register wasn't recorded in the current response packet.
                    stayed_same[register] = stayed_same[register] + 1
        else:
            # this is a query packet
            for register in registers:
                stayed_same[register] = stayed_same[register] + 1
    entry = dict()
    for r in registers:
        entry["switches " + r] = switches[r]
        entry["stayed same " + r] = stayed_same[r]
        if entry["switches " + r] > 0:
            entry["average switch freq" + r] = entry["stayed same " + r] / entry["switches " + r]
        else:
            entry["average switch freq" + r] = -1
    temp_df = pd.DataFrame.from_dict(columns=switch_df.columns,
                                     data={'0': [entry[col] for col in switch_df.columns]}, orient='index')
    switch_df = pd.concat([switch_df, temp_df], ignore_index=True)
    if export_to_excel:
        with open(excel_path + "\\" + sheet_name + ".xlsx", 'w'):
            switch_df.to_excel(excel_path + "\\" + sheet_name + ".xlsx")
    else:
        return switch_df


# get processed data, no binning and no scaling!
# use this only for matrix profiles version. in other versions the calculated stats are the same as for the raw data.
# in the matrix profiles versions the entries after the processing change so the statistics do as well.
# find out the standard deviation and number of unique values
def get_stats_after_processing(processed_df, n, export=True, sheet_name="post processing stats"):
    # get the columns of the registers numbers
    regs = processed_df.columns[1: n + 1]
    cols = ["stdev register" + str(r) for r in regs]
    post_processing_df = pd.DataFrame(columns=cols)
    entry = {stdev: None for stdev in cols}
    for r in regs:
        reg_col = processed_df[regs]
        stdev = statistics.stdev(reg_col)
        entry["stdev register" + str(r)] = stdev
    temp_df = pd.DataFrame.from_dict(columns=post_processing_df.columns,
                                     data={'0': [entry[col] for col in post_processing_df.columns]}, orient='index')
    post_processing_df = pd.concat([post_processing_df, temp_df], ignore_index=True)
    if export:
        with open(excel_path + "\\" + sheet_name + ".xlsx", 'w'):
            post_processing_df.to_excel(excel_path + "\\" + sheet_name + ".xlsx")
    else:
        return post_processing_df


# construct the packet payload according to its function code
def make_payload(modbus_dict, src_port):
    code = int(modbus_dict['modbus.func_code'])
    payload = {}
    if code == 3:
        # it's a response packet
        if src_port == plc_port:

            registers_keys = list(modbus_dict.keys())[4::]
            for register_key in registers_keys:
                splitted_key = register_key.split(" ")
                register = splitted_key[1]
                val = splitted_key[-1]
                payload[register] = val

    return payload


def dissect(pkt_dict):
    layers = pkt_dict['_source']['layers']
    time = datetime.strptime(layers['frame']['frame.time'][0: 28], '%b %d, %Y %H:%M:%S.%f')
    src_ip, dst_ip = layers['ip']['ip.src'], layers['ip']['ip.dst']
    tcp_src, tcp_dst = int(layers['tcp']['tcp.srcport']), int(layers['tcp']['tcp.dstport'])
    func_code = int(layers['modbus']['modbus.func_code'])
    payload = make_payload(layers['modbus'], tcp_src)
    return {'time': time, 'dst_ip': dst_ip, 'src_ip': src_ip, 'dst_port': tcp_dst, 'src_port': tcp_src,
            'func_code': func_code, 'payload': payload}


def filter_data(json_files_dir):
    cols = ['time', 'dst_ip', 'src_ip', 'dst_port', 'src_port', 'func_code', 'payload']
    df = pd.DataFrame(columns=cols)
    codes = ['3', '6', '16']
    for filename in os.listdir(json_files_dir):
        print("working on ", filename)
        file = open(json_files_dir + "//" + filename, "r")
        pkt_data = json.load(file)
        filtered_packets = filter(lambda pkt_dict: pkt_dict['_source']['layers']['modbus']['modbus.func_code'] in codes,
                                  pkt_data)
        dissected_packets = map(lambda pkt_dict: dissect(pkt_dict), filtered_packets)
        file_df = pd.DataFrame(dissected_packets)
        df = df.append(file_df)

    df = df.sort_values('time')

    with open(datasets_path + "\\modbus", "wb") as df_file:
        pickle.dump(df, df_file)


def print_pkts(json_files_dir):
    codes = ['3', '6', '16']
    for filename in os.listdir(json_files_dir):
        file = open(json_files_dir + "//" + filename, "r")
        pkt_data = json.load(file)
        filtered_packets = filter(
            lambda pkt_dict: pkt_dict['_source']['layers']['modbus']['modbus.func_code'] == "3" and
                             pkt_dict['_source']['layers']['tcp']['tcp.srcport'] == "502",
            pkt_data)
        lst_flt = list(filtered_packets)
        mapped_pkts = list(map(lambda pkt_dict: pkt_dict['_source']['layers']['modbus'], lst_flt))
        print(mapped_pkts[0])


def load(dir_path, filename):
    with open(dir_path + "\\" + filename, "rb") as file:
        obj = pickle.load(file)
        return obj


def dump(dir_path, filename, obj):
    with open(dir_path + "\\" + filename, "wb") as file:
        pickle.dump(obj, file)


# ---------------------------------------------------------------------------------------------------------------------------#
# helper functions to bin data
def k_means_binning(df, col_name, n_bins):
    data = df[col_name].to_numpy().reshape(-1, 1)
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit(data)
    labeled_data = k_means.transform(data)
    return labeled_data


def equal_width_discretization(df, col_name, n_bins):
    data = df[col_name].to_numpy().reshape(-1, 1)
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit(data)
    labeled_data = k_means.transform(data)
    return labeled_data


def equal_frequency_discretization(df, col_name, n_bins):
    data = df[col_name].to_numpy().reshape(-1, 1)
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit(data)
    labeled_data = k_means.transform(data)
    return labeled_data


# ---------------------------------------------------------------------------------------------------------------------------#
# construct the data. each entry saves the time passed from the last packet and registers values
def process_data_v1(pkt_df, n, binner=None, n_bins=None, entry_func=None, scale=True):
    # using only 1 PLC
    IP = plc

    # filter out the packets that don't involve the most frequently used PLC
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == IP) | (pkt_df['src_ip'] == IP)]

    # get the most frequently used registers by the PLC
    # frequent_regs = get_frequent_registers(plc_pkts, n)

    # frequent_regs is a list of lists ,so we get the list of our PLC which is the only one used
    registers = to_bin
    cols = np.concatenate((['time'], registers))
    avgs = get_avg_vals(plc_pkts, IP, registers)
    last_values = {r: np.nan for r in registers}

    time_vals_df = pd.DataFrame(columns=cols)

    for i in range(1, len(plc_pkts)):
        # entries from the original data frame
        prev = plc_pkts.iloc[i - 1]
        curr = plc_pkts.iloc[i]

        # the new entry
        new = {}

        # previous entry in the constructed data frame
        prev_entry = {}

        if i != 1:
            prev_entry = time_vals_df.iloc[i - 2]

        # inter-arrival time
        delta_t = curr['time'] - prev['time']
        new['time'] = delta_t.total_seconds()

        src_port = curr['src_port']

        new = entry_func(src_port, curr, registers, new, prev_entry, i, avgs, prev, last_values)
        temp_df = pd.DataFrame.from_dict(columns=time_vals_df.columns,
                                         data={'0': [new[col] for col in time_vals_df.columns]}, orient='index')
        time_vals_df = pd.concat([time_vals_df, temp_df], ignore_index=True)

    for reg_num in registers:
        time_vals_df[reg_num] = time_vals_df[reg_num].fillna(time_vals_df[reg_num].mean())
        if binner is not None and reg_num in to_bin:
            time_vals_df[reg_num] = binner(time_vals_df, reg_num, n_bins)
        if scale:
            time_vals_df[reg_num] = scale_col(time_vals_df, reg_num)

    if scale:
        time_vals_df['time'] = scale_col(time_vals_df, 'time')

    return time_vals_df


# consider only the values of the n most used registers of the plc
def process_data_no_time(pkt_df, n, binner=None, bins=None):
    with_time_df = process_data_v1(pkt_df, n, binner, bins, make_entry_v1)
    no_time = with_time_df.drop(['time'], axis=1)
    return no_time


def make_entry_v1(src_port, curr, registers, new, prev_entry, i, avgs, prev, last_values):
    if src_port == plc_port:
        payload = curr['payload']
        updated_regs = payload.keys()
        # set the columns of the registers values received from the packet
        for reg_num in registers:
            if reg_num in updated_regs:
                new[reg_num] = np.float64(payload[reg_num])
            else:
                # the value of this register wasn't recorded in the packet. get the last value known
                if i == 1:
                    new[reg_num] = np.nan
                else:
                    new[reg_num] = np.float64(prev_entry[reg_num])
    else:
        # a query packet
        for reg_num in registers:
            if i == 1:
                new[reg_num] = np.nan
            else:
                new[reg_num] = np.float64(prev_entry[reg_num])

    return new


def make_entry_v2(src_port, curr, registers, new, prev_entry, i, avgs, prev, last_values):
    if src_port == plc_port:
        payload = curr['payload']
        updated_regs = payload.keys()

        # set the columns of the registers values received from the packet. save the change
        for reg_num in registers:
            prev_reg_val = prev['payload'].get(reg_num, np.nan)
            if reg_num in updated_regs and prev_reg_val is not np.nan:
                # we need to calculate the value change from the last known value.
                new[reg_num] = abs(np.float64(payload[reg_num]) - np.float64(prev_reg_val))
                last_values[reg_num] = np.float64(payload[reg_num])
            else:
                avg = avgs[reg_num]
                if payload.get(reg_num, np.nan) is not np.nan:
                    new[reg_num] = abs(avg - np.float64(payload[reg_num]))
                    last_values[reg_num] = np.float64(payload[reg_num])
                elif prev_reg_val is not np.nan:
                    new[reg_num] = abs(avg - np.float64(prev_reg_val))
                elif last_values[reg_num] is not np.nan:
                    new[reg_num] = abs(avg - np.float64(last_values[reg_num]))
                else:
                    new[reg_num] = np.nan

    else:
        # a query packet
        for reg_num in registers:
            if i == 1:
                new[reg_num] = np.nan
            else:
                prev_reg_val = prev['payload'].get(reg_num, np.nan)
                avg = avgs[reg_num]
                if prev_reg_val is not np.nan:
                    new[reg_num] = abs(avg - np.float64(prev_reg_val))
                else:
                    new[reg_num] = abs(avg - np.float64(last_values.get(reg_num, np.nan)))

    return new


# ---------------------------------------------------------------------------------------------------------------------------
# GAVE BAD RESULTS.
# save inter-arrival time, values of registers ,
# time since registers got those values, similarity score to previous known state
def process_data_v2(pkt_df, n, binner=None, n_bins=None, scale=True, abstract=False):
    # using only 1 PLC
    IP = plc

    # filter out the packets that don't involve the most frequently used PLC
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == IP) | (pkt_df['src_ip'] == IP)]

    # get the most frequently used registers by the PLC
    # frequent_regs = get_frequent_registers(pkt_df, n)
    # frequent_regs is a list of lists ,so we get the list of our PLC which is the only one used
    registers = to_bin
    times = ['time_' + str(r_num) for r_num in registers]
    cols = np.concatenate((['time'], registers, times))

    time_vals_df = pd.DataFrame(columns=cols)

    for i in range(1, len(plc_pkts)):
        # entries from the original data frame
        prev = plc_pkts.iloc[i - 1]
        curr = plc_pkts.iloc[i]
        # the new entry
        new = {}

        # previous entry in the constructed data frame
        prev_entry = {}

        if i != 1:
            prev_entry = time_vals_df.iloc[i - 2]

        if abstract:
            for reg in registers:
                new['time_' + reg] = np.nan

        # inter-arrival time
        delta_t = curr['time'] - prev['time']
        new['time'] = delta_t.total_seconds()

        src_port = curr['src_port']

        if src_port == plc_port:
            payload = curr['payload']
            updated_regs = payload.keys()

            # set the columns of the registers values received from the packet
            for reg_num in registers:
                reg_key = 'time_' + str(reg_num)  # key for the time matching the register
                if reg_num in updated_regs:
                    new[reg_num] = np.float64(payload[reg_num])
                    if not abstract:
                        if i != 1 and new[reg_num] == prev_entry[reg_num]:
                            new[reg_key] = prev_entry[reg_key] + new['time']
                        else:
                            # this register was changed in this packet but, we know of the change only now.
                            # so in the time between the last message and this one we didn't see any change
                            if i != 1:
                                prev_entry[reg_key] += new['time']
                            new[reg_key] = 0
                else:
                    # the value of this register wasn't recorded in the packet. get the last value known
                    if i == 1:
                        new[reg_num] = np.nan
                        if not abstract:
                            new[reg_key] = np.nan
                    else:
                        new[reg_num] = np.float64(prev_entry[reg_num])
                        if not abstract:
                            new[reg_key] = prev_entry[reg_key] + new['time']
        else:
            # a query packet
            for reg_num in registers:
                reg_key = 'time_' + str(reg_num)  # key for the time matching the register
                if i == 1:
                    new[reg_num] = np.nan
                    if not abstract:
                        new[reg_key] = np.nan
                else:
                    new[reg_num] = np.float64(prev_entry[reg_num])
                    if not abstract:
                        new[reg_key] = prev_entry[reg_key] + new['time']

        temp_df = pd.DataFrame.from_dict(columns=time_vals_df.columns,
                                         data={'0': [new[col] for col in time_vals_df.columns]}, orient='index')
        time_vals_df = pd.concat([time_vals_df, temp_df], ignore_index=True)
    if abstract:
        # fill missing values for registers and bin. prepare dataframe for the time calculation.
        for reg_num in registers:
            time_vals_df[reg_num] = time_vals_df[reg_num].fillna(time_vals_df[reg_num].mean())
            time_vals_df[reg_num] = binner(time_vals_df, reg_num, n_bins)
        # iterate over dataframe. update the duration of the values in the registers after the values were binned.
        for n in range(len(time_vals_df)):
            curr = time_vals_df.iloc[n]
            # first packet may have times which are nans.
            if n == 0:
                for reg in registers:
                    reg_key = 'time_' + reg
                    if curr[reg_key] is np.nan:
                        time_vals_df.iloc[0, time_vals_df.columns.get_loc(reg_key)] = 0
            else:
                prev = time_vals_df.iloc[n - 1]
                for reg in registers:
                    curr_reg = curr[reg]
                    prev_reg = prev[reg]
                    register_col_num = time_vals_df.columns.get_loc('time_' + reg)
                    if curr_reg == prev_reg:
                        time_vals_df.iloc[n, register_col_num] += curr['time']
                    else:
                        time_vals_df.iloc[n, register_col_num] = 0
                        time_vals_df.iloc[n - 1, register_col_num] += curr['time']
        for reg_num in registers:
            reg_key = 'time_' + str(reg_num)
            # fill missing values for times.
            time_vals_df[reg_key] = time_vals_df[reg_key].fillna(time_vals_df[reg_key].mean())
            if scale:
                time_vals_df[reg_num] = scale_col(time_vals_df, reg_num)
                time_vals_df[reg_key] = scale_col(time_vals_df, reg_key)
    else:
        for reg_num in registers:
            reg_key = 'time_' + str(reg_num)
            time_vals_df[reg_num] = time_vals_df[reg_num].fillna(time_vals_df[reg_num].mean())
            time_vals_df[reg_key] = time_vals_df[reg_key].fillna(time_vals_df[reg_key].mean())
            if binner is not None:
                time_vals_df[reg_num] = binner(time_vals_df, reg_num, n_bins)
            if scale:
                time_vals_df[reg_num] = scale_col(time_vals_df, reg_num)
                time_vals_df[reg_key] = scale_col(time_vals_df, reg_key)

    if scale:
        time_vals_df['time'] = scale_col(time_vals_df, 'time')

    return time_vals_df


# ---------------------------------------------------------------------------------------------------------------------------
# save inter-arrival time, values of registers ,time being in this state,
# number of packets received while being in this state and the similarity score to previous known state
def process_data_v3(pkt_df, n, binner=None, n_bins=None, scale=True, frequent_vals=None):
    # using only 1 PLC
    IP = plc

    # filter out the packets that don't involve the most frequently used PLC
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == IP) | (pkt_df['src_ip'] == IP)]

    # get the most frequently used registers by the PLC
    # frequent_regs = get_frequent_registers(plc_pkts, n)
    # frequent_regs is a list of lists ,so we get the list of our PLC which is the only one used
    # registers = frequent_regs[IP]
    registers = to_bin
    regs_copy = registers.copy()
    cols = np.concatenate((['time', 'time_in_state'], regs_copy))
    # ['time', 'similarity', 'time_in_state', 'msgs_in_state']
    time_vals_df = pd.DataFrame(columns=cols)

    for i in range(1, len(plc_pkts)):
        # entries from the original data frame
        prev = plc_pkts.iloc[i - 1]
        curr = plc_pkts.iloc[i]
        similarity = 0
        # the new entry
        new = {}

        # previous entry in the constructed data frame
        prev_entry = {}
        df_len = len(time_vals_df)
        if df_len > 0:
            prev_entry = time_vals_df.iloc[df_len - 1]

        # inter-arrival time
        delta_t = curr['time'] - prev['time']
        new['time'] = delta_t.total_seconds()

        src_port = curr['src_port']

        # init columns
        if i == 1:
            new['time_in_state'] = 0

        # a response packet
        if src_port == plc_port:
            payload = curr['payload']
            updated_regs = payload.keys()

            # set the columns of the registers values received from the packet and calculate similarity
            for reg_num in registers:
                if reg_num in updated_regs:
                    new[reg_num] = np.float64(payload[reg_num])
                    # compare to the packet before and update similarity accordingly
                    prev_reg = prev_entry.get(reg_num, np.nan)
                    if df_len != 0 and new[reg_num] == prev_reg:
                        similarity += 1
                else:
                    # the value of this register wasn't recorded in the packet. get the last value known
                    if df_len == 0:
                        new[reg_num] = np.nan
                        if frequent_vals is not None:
                            new[reg_num] = np.float64(frequent_vals[reg_num])
                    else:
                        new[reg_num] = np.float64(prev_entry[reg_num])
                        # no known value change
                        similarity += 1

            # decide if we need to add another entry.
            if df_len != 0:
                similarity /= len(registers)
            else:
                similarity = np.nan

            if similarity == 1:
                time_vals_df.iloc[df_len - 1, 1] += new['time']
            else:
                new['time_in_state'] = 0

                if df_len > 0:
                    time_vals_df.iloc[df_len - 1, 1] += new['time']
                temp_df = pd.DataFrame.from_dict(columns=time_vals_df.columns,
                                                 data={'0': [new[col] for col in time_vals_df.columns]},
                                                 orient='index')
                time_vals_df = pd.concat([time_vals_df, temp_df], ignore_index=True)
        else:
            # a query packet
            if df_len == 0:
                for reg_num in registers:
                    new[reg_num] = np.nan
                    if frequent_vals is not None:
                        new[reg_num] = np.float64(frequent_vals[reg_num])

                new['time_in_state'] = new['time']  # due to the same reason below
                temp_df = pd.DataFrame.from_dict(columns=time_vals_df.columns,
                                                 data={'0': [new[col] for col in time_vals_df.columns]},
                                                 orient='index')
                time_vals_df = pd.concat([time_vals_df, temp_df], ignore_index=True)
            else:
                # there was no state change. so we got 1 more packet in the same state and stayed longer in it
                time_vals_df.iloc[df_len - 1, 1] += new['time']

    for reg_num in registers:
        time_vals_df[reg_num] = time_vals_df[reg_num].fillna(time_vals_df[reg_num].mean())
        if binner is not None and reg_num in to_bin:
            time_vals_df[reg_num] = binner(time_vals_df, reg_num, n_bins)
        if scale:
            time_vals_df[reg_num] = scale_col(time_vals_df, reg_num)

    for col_name in ['time', 'time_in_state']:
        time_vals_df[col_name] = time_vals_df[col_name].fillna(time_vals_df[col_name].mean())
        if scale:
            time_vals_df[col_name] = scale_col(time_vals_df, col_name)

    return time_vals_df


# this data processing is similar to v3 but adds an entry to the dataframe everytime
def process_data_v3_2(pkt_df, n, binner=None, n_bins=None, scale=True, abstract=False):
    # using only 1 PLC
    IP = plc

    # filter out the packets that don't involve the most frequently used PLC
    plc_pkts = pkt_df.loc[(pkt_df['dst_ip'] == IP) | (pkt_df['src_ip'] == IP)]

    # get the most frequently used registers by the PLC
    # frequent_regs = get_frequent_registers(plc_pkts, n)
    # frequent_regs is a list of lists ,so we get the list of our PLC which is the only one used
    registers = to_bin
    regs_copy = registers.copy()
    cols = np.concatenate((['time', 'time_in_state'], regs_copy))

    time_vals_df = pd.DataFrame(columns=cols)

    for i in range(1, len(plc_pkts)):
        # entries from the original data frame
        prev = plc_pkts.iloc[i - 1]
        curr = plc_pkts.iloc[i]
        similarity = 0
        # the new entry
        new = {}

        # previous entry in the constructed data frame
        prev_entry = {}
        df_len = len(time_vals_df)
        if df_len > 0:
            prev_entry = time_vals_df.iloc[df_len - 1]

        # inter-arrival time
        delta_t = curr['time'] - prev['time']
        new['time'] = delta_t.total_seconds()

        src_port = curr['src_port']

        # init columns
        if i == 1:
            new['time_in_state'] = 0
        elif abstract:
            new['time_in_state'] = np.nan

        # a response packet
        if src_port == plc_port:
            payload = curr['payload']
            updated_regs = payload.keys()

            # set the columns of the registers values received from the packet and calculate similarity
            for reg_num in registers:
                if reg_num in updated_regs:
                    new[reg_num] = np.float64(payload[reg_num])
                    # compare to the packet before and update similarity accordingly
                    prev_reg = prev_entry.get(reg_num, np.nan)
                    if df_len != 0 and new[reg_num] == prev_reg:
                        similarity += 1
                else:
                    # the value of this register wasn't recorded in the packet. get the last value known
                    if df_len == 0:
                        new[reg_num] = np.nan
                    else:
                        new[reg_num] = np.float64(prev_entry[reg_num])
                    # no known value change
                    similarity += 1

                similarity /= len(registers)
            if not abstract:
                if similarity == 1 and df_len > 0:
                    # we stayed in the stayed for some more time
                    new['time_in_state'] = (time_vals_df.iloc[df_len - 1, 1] + new['time'])
                else:
                    new['time_in_state'] = 0
                    # if this holds than df_len is 0
                    if similarity == 1:
                        new['time_in_state'] = new['time']
                    # if this holds than similarity is different from 1, still need to update the time in state to account for the time until
                    # the arrival of the current packet
                    if df_len > 0:
                        time_vals_df.iloc[df_len - 1, 1] += new['time']
        else:
            # a query packet
            if df_len == 0:
                for reg_num in registers:
                    new[reg_num] = np.nan

                new['time_in_state'] = new['time']  # this is a query packet so, we don't know about a state change
            else:
                # there was no state change.
                if not abstract:
                    new['time_in_state'] = (time_vals_df.iloc[df_len - 1, 1] + new['time'])
                for reg_num in registers:
                    new[reg_num] = prev_entry[reg_num]

        # in any case we add the new entry to the dataframe.
        temp_df = pd.DataFrame.from_dict(columns=time_vals_df.columns,
                                         data={'0': [new[col] for col in time_vals_df.columns]}, orient='index')
        time_vals_df = pd.concat([time_vals_df, temp_df], ignore_index=True)

    for reg_num in registers:
        time_vals_df[reg_num] = time_vals_df[reg_num].fillna(time_vals_df[reg_num].mean())
        if binner is not None and reg_num in to_bin:
            time_vals_df[reg_num] = binner(time_vals_df, reg_num, n_bins)
        if scale:
            time_vals_df[reg_num] = scale_col(time_vals_df, reg_num)

    if abstract:
        # now we calculate the time_in_state after the temporal abstraction of registers values by binning.
        for n in range(1, len(time_vals_df)):
            curr = time_vals_df.iloc[n]
            prev = time_vals_df.iloc[n - 1]
            similarity = 0
            for reg in registers:
                curr_reg = curr[reg]
                prev_reg = prev[reg]
                if curr_reg == prev_reg:
                    similarity += 1
            similarity /= len(registers)
            if similarity == 1:
                time_vals_df.iloc[n, 1] = curr['time'] + time_vals_df.iloc[n - 1, 1]
            else:
                time_vals_df.iloc[n - 1, 1] += curr['time']
                time_vals_df.iloc[n, 1] = 0

    for col_name in ['time', 'time_in_state']:
        time_vals_df[col_name] = time_vals_df[col_name].fillna(time_vals_df[col_name].mean())
        if scale:
            time_vals_df[col_name] = scale_col(time_vals_df, col_name)

    return time_vals_df


# inter-arrival time,
# time in state so far / value duration of each register,
# state-switching time: upper and lower limit in the neighborhood,
# registers delta from previous packet.
# ----------------------------------------------------- #
# [d1, d2, d3, t-state, ss-upper, ss-lower, inter-arrival time]
# ----------------------------------------------------- #
# in another version : use matrix profiles on the inter-arrival times in some window, get average on the distance vector.
# can do the state switches for single registers
# ----------------------------------------------------- #
# this version has packets of the form:
# [d1, d2, d3, t1, t2, t3, ss-upper, ss-lower, inter-arrival time]
def embedding_v1(pkt_df, neighborhood=20, regs_times_maker=None, binner=None, n_bins=None, scale=True,
                 state_duration=False, matrix_profiles=False, w=-1, j=-1):
    IP = plc
    # plc packets
    plc_pkts = pkt_df.loc[(pkt_df['src_ip'] == IP) | (pkt_df['dst_ip'] == IP)]

    registers = to_bin
    registers_times = ['time_' + reg for reg in registers]
    if not state_duration:
        cols = np.concatenate(
            (['time', 'state_switch_max', 'state_switch_min'], registers, registers_times))
    else:
        cols = np.concatenate(
            (['time', 'state_switch_max', 'state_switch_min', 'time_in_state'], registers))
    if matrix_profiles:
        mp_len = neighborhood - w + 1
        cols = np.concatenate(([cols, ['mp_time_' + str(i) for i in range(mp_len)]]))
    embedded_df = pd.DataFrame(columns=cols)

    avgs = get_avg_vals(plc_pkts, IP, registers)
    last_values = {r: np.nan for r in registers}

    # initialize last_values, there is a neighborhood that we don't process
    for i in range(neighborhood):
        curr = plc_pkts.iloc[i]
        src_port = curr['src_port']
        if src_port == plc_port:
            payload = curr['payload']
            changed = payload.keys()
            # update last value known for the registers if they are in the data payload
            for r in registers:
                if r in changed:
                    last_values[r] = np.float64(payload[r])
    # iterate over the dataframe, look at 20 packets and embed the 21st packet. slide one packet forward
    for i in range(neighborhood, len(plc_pkts)):
        # neighborhood previous packets
        prev_pkts = plc_pkts.iloc[i - neighborhood:i, :]
        curr_pkt = plc_pkts.iloc[i]
        prev_entry = {}

        if i != neighborhood:
            prev_entry = embedded_df.iloc[i - (neighborhood + 1)]

        # build the new df entry
        new = dict()
        time = (curr_pkt['time'] - plc_pkts.iloc[i - 1]['time']).total_seconds()
        new['time'] = time

        src_port = curr_pkt['src_port']

        # we need to initialize the value durations
        # this is different from v2. In v2 we know the values in the previous packet and their duration
        # here we know the delta of the value(in some versions). So, we need to keep track of the durations and update them
        # when processing new packets. this checks for how long the last value is in the register.
        durations = {r: np.nan for r in registers}
        if i == neighborhood:
            p0 = plc_pkts.iloc[0]
            if p0['src_port'] == plc_port:
                payload = p0['payload']
                for r in payload.keys():
                    if r in registers:
                        durations[r] = 0
            # forward iteration.
            for j in range(1, neighborhood):
                p = plc_pkts.iloc[j]
                p_prev = plc_pkts.iloc[j - 1]
                inter_arrival = (p['time'] - p_prev['time']).total_seconds()

                src_port = p['src_port']

                # query , increase durations
                if src_port != plc_port:
                    for r in durations.keys():
                        # we need durations[r] to be not nan. If it's nan then we don't accumulate the time for that register
                        # yet.
                        if durations[r] is not np.nan:
                            durations[r] = durations[r] + inter_arrival
                else:
                    # response
                    payload = p['payload']
                    changed = payload.keys()
                    # If last_values[r] is np.nan for some register than we will never encounter it in some
                    # packet payload and the duration will stay np.nan. So, it's sufficient to consider only cases where
                    # last_values[r] is not np.nan.
                    for r in durations.keys():
                        # we know of a value for that register and it's in the payload
                        if last_values[r] is not np.nan and r in changed:
                            last_value = last_values[r]
                            payload_value = np.float64(payload[r])
                            # if they are the same.
                            if last_value == payload_value:
                                if durations[r] is np.nan:  # first time seeing a value for the register.
                                    durations[r] = 0
                                # same values so increase duration of value.
                                durations[r] += inter_arrival
                            # if we see a different value then we know of a duration of 0 for now.
                            if last_value != payload_value:
                                durations[r] = 0
                        # we know of a value for the register but not seen in this packet
                        elif last_values[r] is not np.nan:
                            # if the duration is nan it means that the register never got a value so far.
                            # so we don't inc any duration. If it got some value, it means that the duration isn't nan.
                            # If the value it got is the last_value we will inc the time and get the correct duration.
                            # However, if it's not the last value than in some future packet we will see the last value,
                            # 0 the duration and be in the first case and get the correct duration.
                            if durations[r] is not np.nan:
                                durations[r] += inter_arrival
                        # in all other cases last_values[r] is nan so we don't consider them.

        regs_times_maker(src_port, curr_pkt, last_values, neighborhood, i, durations, new, prev_entry,
                         time, avgs, plc_pkts.iloc[i - 1]['payload'], registers, state_duration, embedded_df)
        # calculate state switches times
        num_state_switches = 0
        time_in_same_state = 0
        state_switch_times = []
        last_known_state = {r: np.nan for r in registers}
        first = prev_pkts.iloc[0]
        if first['src_port'] == plc_port:
            payload = first['payload']
            for r in registers:
                last_known_state[r] = payload.get(r, np.nan)

        for p_i in range(1, len(prev_pkts)):
            pkt = prev_pkts.iloc[p_i]
            prev_pkt = prev_pkts.iloc[p_i - 1]
            t = (pkt['time'] - prev_pkt['time']).total_seconds()
            port = pkt['src_port']

            same = True
            if port == plc_port:
                # response, check that state is the same
                payload = pkt['payload']
                for reg in registers:
                    if reg in payload.keys():
                        val = payload[reg]
                        if val != last_known_state[reg]:
                            same = False
                            last_known_state[reg] = payload[reg]
            if same:
                time_in_same_state += t
            else:
                num_state_switches += 1
                state_switch_times.append(time_in_same_state)
                time_in_same_state = 0
        if num_state_switches == 0:
            mean_state_switch_time = np.nan
        else:
            mean_state_switch_time = statistics.mean(state_switch_times)
        if num_state_switches > 1:
            stdev_state_switch_time = statistics.stdev(state_switch_times)
        else:
            stdev_state_switch_time = 0
        state_switch_upper = mean_state_switch_time + 3 * stdev_state_switch_time
        state_switch_lower = mean_state_switch_time - 3 * stdev_state_switch_time
        new['state_switch_max'] = state_switch_upper
        new['state_switch_min'] = state_switch_lower
        if matrix_profiles:
            times_data = np.array([], dtype=float)
            for p_i in range(neighborhood - 1):
                p_p = prev_pkts.iloc[p_i]
                p_c = prev_pkts.iloc[p_i + 1]
                secs = float((p_c['time'] - p_p['time']).total_seconds())
                np.append(times_data, secs)
            np.append(times_data,
                      float((curr_pkt['time'] - (prev_pkts.iloc[neighborhood - 1])['time']).total_seconds()))
            times_df = pd.DataFrame(columns=['time'], data=times_data)
            mp_df = matrix_profiles_pre_processing(times_df, neighborhood, w, j, np.argmin)
            for j in range(neighborhood - w + 1):
                new['mp_time_' + str(j)] = mp_df.iloc[j]
        temp_df = pd.DataFrame.from_dict(columns=cols, data={'0': [new[c] for c in cols]}, orient='index')
        embedded_df = pd.concat([embedded_df, temp_df], axis=0, ignore_index=True)

    # fill missing values and bin
    for reg_num in registers:
        reg_key = 'time_' + reg_num
        embedded_df[reg_num] = embedded_df[reg_num].fillna(embedded_df[reg_num].mean())
        if not state_duration:
            embedded_df[reg_key] = embedded_df[reg_key].fillna(embedded_df[reg_key].mean())
        if binner is not None:
            embedded_df[reg_num] = binner(embedded_df, reg_num, n_bins)
        if scale:
            embedded_df[reg_num] = scale_col(embedded_df, reg_num)
            if not state_duration:
                embedded_df[reg_key] = scale_col(embedded_df, reg_key)
    # scale
    if scale:
        cs = ['time', 'state_switch_max', 'state_switch_min']
        if state_duration:
            cs.append('time_in_state')
        if matrix_profiles:
            mp_len = neighborhood - w + 1
            for i in range(mp_len):
                cs.append('mp_time_' + str(i))
        for c in cs:
            embedded_df[c] = embedded_df[c].fillna(embedded_df[c].mean())
            embedded_df[c] = scale_col(embedded_df, c)

    return embedded_df


def embed_v1_with_deltas_regs_times(src_port, curr_pkt, last_values, neighborhood, i, durations, new, prev_entry, time,
                                    avgs, prev_payload, registers, state_duration, embedded_df):
    # a response packet
    if src_port == plc_port:
        payload = curr_pkt['payload']
        changed = payload.keys()
        for r in registers:
            # we saw a value before and we see a value in the current payload
            if last_values[r] is not np.nan and r in changed:
                delta = np.float64(np.abs(last_values[r] - np.float64(payload[r])))
                # different value
                if last_values[r] != np.float64(payload[r]):
                    last_values[r] = np.float64(payload[r])
                    durations[r] = 0
                    # we know of the change only now, if possible update the duration of the previous value
                    if not state_duration:
                        new['time_' + r] = 0
                        if i != neighborhood:
                            embedded_df.iloc[i - (neighborhood + 1), embedded_df.columns.get_loc('time_' + r)] += time
                else:
                    # same value as the last known value, increase duration and that's it
                    durations[r] += time
                    if not state_duration:
                        new['time_' + r] = durations[r]
            # first time we see a value for the register
            elif last_values[r] is np.nan and r in changed:
                delta = np.float64(np.abs(avgs[r] - np.float64(payload[r])))
                last_values[r] = np.float64(payload[r])
                durations[r] = 0
                if not state_duration:
                    new['time_' + r] = 0
            # if we get here it means that r is not in changed and we never saw a value for it
            elif last_values[r] is np.nan:
                delta = np.nan
                if not state_duration:
                    new['time_' + r] = np.nan
            # if we get here it means that r is not in changed and that last_values[r] != nan
            # inc duration of the last known value
            else:
                delta = np.float64(np.abs(avgs[r] - last_values[r]))
                durations[r] += time
                if not state_duration:
                    new['time_' + r] = durations[r]
            new[r] = delta
            if state_duration:
                nans = np.isnan(np.array(list(durations.values()), dtype=np.float64))
                if np.alltrue(nans):
                    new['time_in_state'] = np.nan
                else:
                    new['time_in_state'] = min(durations.values())
    # a query packet
    else:
        for reg in registers:
            if durations[reg] is np.nan:
                durations[reg] = 0
            durations[reg] += time
            if not state_duration:
                new['time_' + reg] = durations[reg]
            prev_delta_reg = prev_payload.get(reg, np.NaN)
            if prev_delta_reg is not np.NaN:
                new[reg] = np.float64(np.abs(np.float64(prev_delta_reg) - avgs[reg]))
            else:
                new[reg] = np.float64(np.abs(last_values[reg] - avgs[reg]))
        if state_duration:
            nans = np.isnan(np.array(list(durations.values()), dtype=np.float64))
            if np.alltrue(nans):
                new['time_in_state'] = np.nan
            else:
                new['time_in_state'] = min(durations.values())


def embed_v1_with_values_regs_times(src_port, curr_pkt, last_values, neighborhood, i, durations, new, prev_entry, time,
                                    avgs, prev_payload, registers, state_duration, embedded_df):
    # a response packet
    if src_port == plc_port:
        payload = curr_pkt['payload']
        changed = payload.keys()
        for r in registers:
            # we see a value in the current payload
            if r in changed and last_values[r] is not np.nan:
                new[r] = np.float64(payload[r])
                # different value
                if last_values[r] != payload[r]:
                    last_values[r] = np.float64(payload[r])
                    durations[r] = 0
                    # we know of the change only now, if possible update the duration of the previous value
                    if not state_duration:
                        new['time_' + r] = 0
                        if i != neighborhood:
                            embedded_df.iloc[i - (neighborhood + 1), embedded_df.columns.get_loc('time_' + r)] += time
                else:
                    # same value as the last known value, increase duration and that's it
                    durations[r] += time
                    if not state_duration:
                        new['time_' + r] = durations[r]
            # first time we see a value for the register
            elif last_values[r] is np.nan and r in changed:
                new[r] = np.float64(payload[r])
                if not state_duration:
                    new['time_' + r] = 0
                last_values[r] = np.float64(payload[r])
                durations[r] = 0
            # if we get here it means that r is not in changed and we never saw a value for it
            elif last_values[r] is np.nan:
                new[r] = np.nan
                if not state_duration:
                    new['time_' + r] = np.nan  # decide how to mix this update with the prev_entry update
            # if we get here it means that r is not in changed and that last_values[r] != nan
            # inc duration of the last known value
            else:
                new[r] = last_values[r]
                durations[r] += time
                if not state_duration:
                    new['time_' + r] = durations[r]
        if state_duration:
            nans = np.isnan(np.array(list(durations.values()), dtype=np.float64))
            if np.alltrue(nans):
                new['time_in_state'] = np.nan
            else:
                new['time_in_state'] = min(durations.values())
    # a query packet
    else:
        for reg in registers:
            if durations[reg] is np.nan:
                durations[reg] = 0
            durations[reg] += time
            if not state_duration:
                new['time_' + reg] = durations[reg]
            new[reg] = prev_entry[reg]
    if state_duration:
        nans = np.isnan(np.array(list(durations.values()), dtype=np.float64))
        if np.alltrue(nans):
            new['time_in_state'] = np.nan
        else:
            new['time_in_state'] = min(durations.values())


def grid_search_binning():
    binners = [k_means_binning, equal_frequency_discretization, equal_width_discretization]
    names = {k_means_binning: "k_means", equal_frequency_discretization: "equal_frequency",
             equal_width_discretization: "equal_width"}
    n_bins = [5, 6, 7, 8, 9, 10]
    pkt_df = load(datasets_path, "modbus")
    options = itertools.product(binners, n_bins)
    for binner_bins in options:
        binner = binner_bins[0]
        bins = binner_bins[1]
        model_name = "v1_single_plc_make_entry_v1_20_packets_MP_min_{}_{}".format(names[binner], bins)
        processed_df = process_data_v1(pkt_df, 5, binner, bins, make_entry_v1)
        models.matrix_profiles_LSTM(processed_df, 20, 10, 10, 42, model_name)


def compare_models(models_folder, metric, metric_name, plot_name):
    plt.clf()
    bins = [5, 6, 7, 8, 9, 10]
    scores = []
    print("plotting")
    for model_folder in os.listdir(modeles_path + "\\" + models_folder):
        model_path = modeles_path + "\\" + models_folder + "\\" + model_folder
        model = keras.models.load_model(model_path)
        X_test = load(datasets_path + "\\" + models_folder, "X_test_" + model_folder)
        y_test = load(datasets_path + "\\" + models_folder, "y_test_" + model_folder)
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        scores.append(score)
    plt.scatter(bins, scores)
    plt.title(plot_name)
    plt.xlabel("number of bins")
    plt.ylabel(metric_name)
    with open(plots_path + "\\" + plot_name, "w"):
        plt.savefig(plots_path + "\\" + plot_name + " plot.png")


# we take a series and brake it into sub-series and look at the "distance" between the subseries and
# the whole series. given a series of matrix profiles processed "packets" the next element which we will try to
# predict is the difference of the first sub-series from the series in the next "whole series".
# that way we look at the relation between a series of packets and another series of packets and not a series of packets to a single packet
# ASSUMPTION: series_len is an even number
def matrix_profiles_pre_processing(pkt_data, series_len, window, jump, index_finder):
    # convert columns to float for MASS algorithm
    float_df = pkt_data
    for col in float_df.columns:
        float_df[col] = float_df[col].astype(float)

    # get distance vector of a single series
    def calc_dv(series, window):
        j = 0
        series_dv = pd.DataFrame(columns=series.columns)

        # while we can take another window
        while j < len(series) - window + 1:
            curr_w = series.iloc[j: j + window]
            curr_dv = pd.DataFrame(columns=series.columns)

            # get distance vector for each column
            for col in curr_w.columns:
                curr_dv_col = stumpy.mass(curr_w[col], series[col])
                curr_dv[col] = curr_dv_col

            starting_index = j
            exclusion_start = max(0, starting_index - (window / 2))
            exclusion_end = starting_index + (window / 2)
            dv_low = curr_dv.iloc[:int(exclusion_start)]
            dv = dv_low

            if exclusion_end < len(curr_dv):
                dv_high = curr_dv.iloc[int(exclusion_end):]
                dv = pd.concat([dv_low, dv_high], axis=0, ignore_index=True)

            # look at the distances across all columns. get the row with the min/max distance considering all columns.
            row_dists = [np.sum(np.square(dv.iloc[i].to_numpy())) for i in range(0, len(dv))]
            chosen_dist = dv.iloc[index_finder(row_dists)]
            chosen_dist_dict = {}
            for i in range(len(series_dv.columns)):
                c = series_dv.columns[i]
                chosen_dist_dict[c] = chosen_dist[i]
            temp_df = pd.DataFrame.from_dict(data={'0': [chosen_dist_dict[c] for c in series_dv.columns]},
                                             columns=series_dv.columns, orient='index')
            series_dv = pd.concat([series_dv, temp_df], axis=0, ignore_index=True)
            # slide
            j += 1

        return series_dv

    i = 0
    dvs = pd.DataFrame(columns=pkt_data.columns)

    # while we can still take another series
    while i < len(float_df) - series_len + 1:
        curr_series = float_df.iloc[i: i + series_len]
        dv = calc_dv(curr_series, window)
        dvs = pd.concat([dvs, dv], axis=0, ignore_index=True)
        # jump
        i += jump
    return dvs


# process the raw data using some method without binning but with scaling. Then split the data, convert to csv and save.
# The csv file is formatted by the requirements of HTM.
def create_data_for_HTM():
    binners = [k_means_binning, equal_frequency_discretization, equal_width_discretization]
    names = {k_means_binning: "k_means", equal_frequency_discretization: "equal_frequency",
             equal_width_discretization: "equal_width"}
    n_bins = [5, 6, 7, 8, 9, 10]
    data_version = 'v1_1'
    pkt_df = load(datasets_path, "modbus")
    options = itertools.product(binners, n_bins)
    for binner_bins in options:
        binner = binner_bins[0]
        bins = binner_bins[1]
        processed_df = process_data_v1(pkt_df, 5, binner=None, n_bins=None, entry_func=make_entry_v1, scale=True)
        # X_train will be used to train the HTM network. X_test and sets created by injecting anomalies into X_test will be used
        # for testing the HTM network.
        X_train, X_test = train_test_split(processed_df, test_size=0.2, random_state=42)
        # 1. write column names.
        # 2. write columns data types.
        # 3. write df to csv without the columns names.
        folder = datasets_path + '\\' + 'HTM_{}_{}'.format(names[binner], data_version)

        train_path_str = folder + '\\' + "X_train_single_plc_v1_1_HTM_{}_{}.csv".format(names[binner], bins)
        test_path_str = folder + '\\' + "X_test_single_plc_v1_1_HTM_{}_{}.csv".format(names[binner], bins)
        train_path = Path(train_path_str)
        test_path = Path(test_path_str)

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        with open(train_path_str, 'w', newline='') as train_file:
            train_writer = csv.writer(train_file)
            # write the field names.
            train_cols = list(X_train.columns)
            train_writer.writerow(train_cols)
            # write the field types.
            train_cols_types = ['float'] * len(train_cols)
            train_writer.writerow(train_cols_types)
            # use no flags.
            train_writer.writerow([])
        X_train.to_csv(path_or_buf=train_path, index=False, header=False, mode='a')

        with open(test_path_str, 'w', newline='') as test_file:
            test_writer = csv.writer(test_file)
            # write the field names.
            test_cols = list(X_test.columns)
            test_writer.writerow(test_cols)
            # write the field types.
            test_cols_types = ['float'] * len(test_cols)
            test_writer.writerow(test_cols_types)
            # use no flags.
            test_writer.writerow([])
        X_test.to_csv(path_or_buf=test_path, index=False, header=False, mode='a')


# ---------------------------------------------------------------------------------------------------------------------------
# bring the data to excel, used to analyze the performance of regressors.
def export_results(models_folder, columns, sheet_name, data_version, series_length, binning, pred_len=1, layer=1,
                   s=None, w=None, j=None):
    print("working...")
    regular_cols = ['data_version', 'series length', 'binning', 'n_bins', 'mse', 'r2',
                    'prediction length', 'number of layers']
    mp_cols = np.concatenate((regular_cols, ['MP series', 'MP window', 'MP jump']))
    comparisons_file = excel_path + "\\" + sheet_name + ".xlsx"
    if s is None:
        results_df = pd.DataFrame(columns=regular_cols)
    else:
        results_df = pd.DataFrame(
            columns=mp_cols)
    for model_folder in os.listdir(modeles_path + '\\' + models_folder):
        model_path = modeles_path + "\\" + models_folder + "\\" + model_folder
        X_test = load(datasets_path + "\\" + models_folder, "X_test_" + model_folder)
        y_test = load(datasets_path + "\\" + models_folder, "y_test_" + model_folder)
        model = keras.models.load_model(model_path)
        y_pred = model.predict(X_test)
        r2 = 0
        mse = 0
        if pred_len != 1:
            for i in range(pred_len):
                y_p = [s[i] for s in y_pred]
                y_t = [s[i] for s in y_test]
                r2 += r2_score(y_t, y_p)
                mse += mean_squared_error(y_t, y_p)
            r2 /= pred_len
            mse /= pred_len
        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        if pred_len == 1:
            """y_pred_df = pd.DataFrame(y_pred, columns=columns)
             y_true_df = pd.DataFrame(y_test, columns=columns)

            excel_df = pd.DataFrame()
            for col in y_true_df.columns:
                true_col = 'true' + col
                pred_col = 'pred' + col
                excel_df[true_col] = y_true_df[col]
                excel_df[pred_col] = y_pred_df[col]

            # with open(excel_path + "\\data_" + model_folder, 'w'):
            # excel_df.to_excel(excel_path + "\\data_" + model_folder + ".xlsx")"""
        splitted = model_folder.split(sep='_')
        bins = int(splitted[-1])
        if s is None:
            result_df = pd.DataFrame([[data_version, series_length, binning, bins, mse, r2, pred_len, layer]],
                                     columns=regular_cols)
        else:
            result_df = pd.DataFrame([[data_version, series_length, binning, bins, mse, r2, pred_len, layer, s, w, j]],
                                     columns=mp_cols)
        results_df = pd.concat([results_df, result_df], ignore_index=True, axis=0)

    with pd.ExcelWriter(comparisons_file, mode='w') as writer:
        results_df.to_excel(writer, sheet_name=sheet_name)


if __name__ == '__main__':
    print('Hello !')
