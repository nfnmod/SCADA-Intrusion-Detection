# this class is responsible for preparing the input for the KarmaLego algorithm.
import copy
import csv
import itertools
import os.path
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

import data

payload_col_number = 6

KL_symbols = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL symbols'
KL_entities = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL entities'
KL_events = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL events'
binners_base = '//sise//home//zaslavsm//SCADA//binners'
folder_name = 'v1_1'
file_name = 'v1_single_plc_make_entry_v1_20_packets'


# ---------------------------------------------------------------------------------------------------------------------------#
# helper functions to bin data
def k_means_binning(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))[0]


def equal_width_discretization(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))[0]


def equal_frequency_discretization(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))[0]


# ---------------------------------------------------------------------------------------------------------------------------#
# the function will leave in the payload only the registers which change their values
def process_payload(df, regs_to_use):
    new_payloads = []
    for i in range(len(df)):
        pkt = df.iloc[i]
        src = pkt['src_port']
        new_payload = {}
        if src == data.plc_port:
            payload = pkt['payload']
            new_payload = {}
            for reg in payload.keys():
                if reg in regs_to_use:
                    new_payload[reg] = payload[reg]
        new_payloads.append(new_payload)

    # done processing this payload, now switch the original and the new one.
    c_df = df.copy()
    c_df.loc[:, 'payload'] = new_payloads
    return c_df


def define_entities(df_data):
    """
    :param df_data: holds the data itself.
    :return:
    """
    df = df_data
    entity_counter = 0
    entities = {}
    for i in range(len(df)):
        packet = df.iloc[i]
        # if it's a packet originating from the HMI then it has no payload since we only consider read commands.
        # so there is a payload only in packets which are response messages.
        if packet['src_port'] == data.plc_port:
            IP = packet['src_ip']
            payload = packet['payload']
            # FORCE ORDER HERE.
            payload_registers = [int(reg) for reg in sorted(payload.keys())]
            for register in payload_registers:
                entity = (IP, register,)  # the entity id of the register of the plc.
                existing_entity = entities.get(entity, None)  # check if it got an id already.
                if existing_entity is None:
                    entities[entity] = entity_counter  # set the id
                    entity_counter += 1  # update the id
    return entities


def get_pkt_entities(pkt):
    """

    :param pkt: a response packet
    :return: the entities in the packet.
    """
    IP = pkt['src_ip']
    payload = pkt['payload']
    # entities in the packet.
    packet_entities = set()
    # FORCE ORDER HERE.
    for reg in sorted(payload.keys()):
        packet_entities.add((IP, int(reg),))
    return packet_entities


def load_binners(numbers_of_bins, binning_methods_folders_names, regs):
    binners = dict()
    folders = {'KMeans': 'k_means', 'EqualFreq': 'equal_frequency',
               'EqualWidth': 'equal_width'}
    for number_of_bins in numbers_of_bins:
        for binning_methods_folder_name in binning_methods_folders_names:
            model_name = '{}_{}_{}'.format(file_name, folders[binning_methods_folder_name], number_of_bins)
            suffix = '//{}_{}'.format(binning_methods_folder_name, folder_name) + '//{}'.format(
                model_name)
            for reg in regs:
                with open(binners_base + suffix + '_{}'.format(reg),
                          mode='rb') as binner_p:
                    binner = pickle.load(binner_p)
                    binners[(number_of_bins, binning_methods_folder_name, reg)] = binner
    return binners


# compute the events in the sliding windows.
def define_events_in_sliding_windows(df, binning_methods, numbers_of_bins, w, regs_to_use, consider_last=True, bin=True,
                                     ready_symbols=None,
                                     ready_entites=None):
    """

    :param ready_entites: same as for ready symbols
    :param ready_symbols: when testing, we get the symbols already discovered in the data set to ensure all the symbols
                          which can be found in the test data set are already defined.
    :param bin: bin or not (used for testing)
    :param regs_to_use: registers to consider
    :param df:
    :param binning_methods: binning methods to use.
    :param numbers_of_bins: numbers of bins to use.
    :param w: window size.
    :param consider_last: make the last event last until the start of the next window.
    :return:
    """
    # n_windows = len(df) - w + 1
    n_windows = 1000
    binning_methods_inv = {k_means_binning: 'KMeans', equal_frequency_discretization: 'EqualFreq',
                           equal_width_discretization: 'EqualWidth'}
    binners = load_binners(numbers_of_bins, binning_methods.keys(), regs_to_use)
    sw_events_single = {sw_num: [] for sw_num in range(n_windows)}
    sw_events_dict = {(binning_methods_inv[b], k): copy.deepcopy(sw_events_single) for (b, k) in
                      itertools.product(binning_methods.values(), numbers_of_bins)}
    folders = {'KMeans': "k_means", 'EqualFreq': 'equal_frequency',
               'EqualWidth': "equal_width"}
    df = process_payload(df, regs_to_use)

    if ready_symbols == dict():
        entities = define_entities(df)
        symbols_single = ready_symbols
        symbol_counter_single = len(symbols_single.keys())

        symbols_counter_dict = {(binning_methods_inv[b], k): symbol_counter_single for (b, k) in
                                itertools.product(binning_methods.values(), numbers_of_bins)}

        entities_dict = {(binning_methods_inv[b], k): entities for (b, k) in
                         itertools.product(binning_methods.values(), numbers_of_bins)}

        symbols_dict = {(binning_methods_inv[b], k): copy.deepcopy(symbols_single) for (b, k) in
                        itertools.product(binning_methods.values(), numbers_of_bins)}
    else:
        entities_dict = ready_entites
        symbols_dict = ready_symbols
        symbols_counter_dict = {(binning_methods_inv[b], k): len(ready_symbols[(binning_methods_inv[b], k)].keys()) for
                                (b, k) in
                                itertools.product(binning_methods.values(), numbers_of_bins)}

    # save for each (entity, binning method, number of bins) -> # window last seen in
    # save for each (entity, binning method, number of bins) -> values and times sequences and idx from the last window seen in
    # save for each (entity, binning method, number of bins) -> sequence of all (idx,value) of packets in the window the entity was seen in
    entities_strs = entities_dict[(binning_methods_inv[list(binning_methods.values())[0]], numbers_of_bins[0])].keys()
    last_window_info_dict = {e: None for e in entities_strs}

    # per window
    prev_entities = []
    for i in range(n_windows):
        # with open('//sise//home//zaslavsm//SCADA//log files//TIRP log.txt', mode='a') as f:
        # f.write('window #{}\n'.format(i))

        checked_entities = []
        window = df.iloc[i: i + w]
        start_time = (window.iloc[0])['time']
        window_entities = set()
        if i == 0:
            for w_i in range(len(window)):
                w_pkt = window.iloc[w_i]
                src = w_pkt['src_port']
                if src == data.plc_port:
                    prev_entities += list(get_pkt_entities(w_pkt))
            window_entities = set(prev_entities)
        else:
            to_remove = list(get_pkt_entities(df.iloc[i - 1]))
            prev_entities = prev_entities[len(to_remove):]
            prev_entities += list(get_pkt_entities(window.iloc[-1]))
            window_entities = set(prev_entities)
        entities_events_single = {e: [] for e in window_entities}
        entities_events_dict = {(binning_methods_inv[b], k): copy.deepcopy(entities_events_single) for (b, k) in
                                itertools.product(binning_methods.values(), numbers_of_bins)}

        for j in range(w):
            pkt = window.iloc[j]
            # we want to look for entities in the packet.
            if pkt['src_port'] == data.plc_port:
                IP = pkt['src_ip']
                payload = pkt['payload']
                # entities in the packet.
                packet_entities = get_pkt_entities(pkt)
                for entity in sorted(packet_entities):
                    # find the values the entity received in the window and bin them.
                    # find the times at which the entity has received the values.
                    # create events and add them to the list of the entity's events.
                    # mark the entity as checked.
                    # no need to consider previous packets because the entity doesn't appear in them. If it were to appear in them
                    # then it wouldn't be unchecked.

                    if entity not in checked_entities:
                        entity_info = last_window_info_dict[entity]
                        reg_num = entity[1]
                        values = [float(payload[str(reg_num)])]
                        first_appearances_curr_window = [j]
                        idx_vals_seq_curr_window = [(j, values[0])]
                        times = [round((pkt['time'] - start_time).total_seconds())]
                        times_wo_round = [(pkt['time'] - start_time).total_seconds()]  # all w.o rounding
                        start_idx = j + 1

                        # if window # 0  -> do full calculation.
                        if entity_info is None:
                            start_idx = j + 1
                        # (#last window - # current window + j) >= window size -> do full calculation.
                        elif (i - entity_info['last_window_num'] + j) >= w:
                            start_idx = j + 1
                        else:
                            # they intersect.

                            window_num_delta = i - entity_info['last_window_num']
                            idx_vals_seq = entity_info['idx,value']
                            first_appearance = idx_vals_seq[0][0] - window_num_delta
                            first_appearances = entity_info['first_appearances_idx']

                            time_delta = (window.iloc[0]['time'] - df.iloc[entity_info['last_window_num']][
                                'time']).total_seconds()

                            if first_appearance in range(w):
                                # first appearance in last window is in this window as well.
                                # take everything from the last one and continue from where it stops.
                                start_idx = idx_vals_seq[-1][0] + 1 - window_num_delta
                                values = entity_info['values']
                                times = [round(t - time_delta) for t in
                                         entity_info['times_wo_round']]  # round(entity_info['times'] - time_delta)
                                # print('window # {}\n no bin search, entity {}'.format(i, entity))
                                first_appearances_curr_window = [first_appearance - window_num_delta for
                                                                 first_appearance in first_appearances]
                                idx_vals_seq_curr_window = [(idx - window_num_delta, val) for (idx, val) in
                                                            idx_vals_seq]
                                times_wo_round = [t - time_delta for t in entity_info['times_wo_round']]
                            else:
                                # take times and values from first idx greater than current one.
                                # subtract the time difference.
                                # set the range to begin from the last index of the idx sequences.

                                # find the first next first appearance of a different value.
                                l = 0
                                r = len(first_appearances)
                                num_fa = len(first_appearances)
                                first_gt = -1

                                while l <= r:
                                    middle = round((l + r) / 2)
                                    idx = first_appearances[middle] - window_num_delta

                                    if idx == j:
                                        first_gt = middle + 1
                                        break
                                    elif idx < j:
                                        l = middle + 1
                                        if first_appearances[middle + 1] > j:
                                            first_gt = middle + 1
                                            break
                                    else:
                                        r = middle - 1
                                        if first_appearances[middle - 1] <= j:
                                            first_gt = middle
                                            break
                                if first_gt == -1 or first_gt >= num_fa:
                                    start_idx = j + 1
                                else:
                                    values += entity_info['values'][first_gt:]
                                    times += [round(t - time_delta) for t in entity_info['times_wo_round'][
                                                                             first_gt:]]  # (round(entity_info['times'][first_gt:] - time_delta))
                                    # print('window # {}\n bin search, entity {}'.format(i, entity))
                                    first_appearances_curr_window += [fa - window_num_delta for fa in
                                                                      first_appearances[first_gt:]]
                                    start_idx = idx_vals_seq[-1][0] + 1 - window_num_delta
                                    times_wo_round += [t - time_delta for t in entity_info['times_wo_round'][first_gt:]]

                                    l = 0
                                    r = len(idx_vals_seq)
                                    from_idx = -1

                                    while l <= r:
                                        middle = round((l + r) / 2)
                                        idx = idx_vals_seq[middle][0] - window_num_delta
                                        if idx == j:
                                            from_idx = middle
                                            break
                                        elif idx < j:
                                            l = middle + 1
                                        else:
                                            r = middle - 1

                                    idx_vals_seq_curr_window = [(idx - window_num_delta, val) for (idx, val) in
                                                                idx_vals_seq[from_idx:]]

                        for k_w in range(start_idx, w):
                            w_pkt = window.iloc[k_w]
                            # check for the entity in the payload.
                            if w_pkt['src_port'] == data.plc_port:
                                w_IP = w_pkt['src_ip']
                                w_reg_val = w_pkt['payload'].get(str(reg_num), None)
                                ip_and_in_payload = (w_IP == IP and w_reg_val is not None)

                                if ip_and_in_payload:
                                    idx_vals_seq_curr_window.append((k_w, float(w_reg_val)))

                                if ip_and_in_payload and (len(values) == 0 or values[-1] != w_reg_val):
                                    values.append(float(w_reg_val))
                                    times.append(round((w_pkt['time'] - start_time).total_seconds()))
                                    times_wo_round.append((w_pkt['time'] - start_time).total_seconds())
                                    first_appearances_curr_window.append(k_w)
                        # mark as checked.
                        checked_entities.append(entity)
                        last_window_info_dict[entity] = {'last_window_num': i,
                                                         'idx,value': idx_vals_seq_curr_window,
                                                         'values': values,
                                                         # 'times': times,
                                                         'time_wo_round': times_wo_round,
                                                         'first_appearances_idx': first_appearances_curr_window,
                                                         'idx_vals_seq_curr_window': first_appearances_curr_window}
                        # bin using all options to avoid reprocessing

                        if bin:
                            for (b, k) in itertools.product(binning_methods.values(), numbers_of_bins):
                                values = np.array(values).reshape(-1, 1)
                                values = (binners[k, binning_methods_inv[b], str(entity[1])].transform(
                                    values)).tolist()  # bin.
                                values = [vlist[0] for vlist in values]

                                symbols = symbols_dict[(binning_methods_inv[b], k)]
                                symbol_counter = symbols_counter_dict[(binning_methods_inv[b], k)]
                                entities_events = entities_events_dict[(binning_methods_inv[b], k)]
                                entities = entities_dict[(binning_methods_inv[b], k)]

                                n = 0
                                if len(values) > 0:
                                    # all but last value-time pair.
                                    while n < len(values) - 1:
                                        value = values[n]
                                        s_t = times[n]
                                        n += 1  # avoid infinite loop if all values are different.
                                        while n < len(values) and value == values[n]:
                                            n += 1
                                        f_t = times[min(n, len(values) - 1)]
                                        symbol = (entities[entity], value,)
                                        if symbols.get(symbol, None) is None:
                                            symbols[symbol] = symbol_counter
                                            symbol_counter += 1
                                            symbols_counter_dict[(binning_methods_inv[b], k)] += 1
                                        event = (s_t, f_t, symbols[symbol],)
                                        entities_events[entity].append(event)
                                    if consider_last and i + w < len(df):
                                        # the last value-time pair. the symbol "holds" until the first packet of the next
                                        v = values[len(values) - 1]
                                        finish = df.iloc[i + w]['time']
                                        if len(values) > 1:
                                            # there have been more than 1 value changes.
                                            # last event
                                            event = entities_events[entity][-1]
                                            event_symbol_number = event[2]
                                            vals = list(symbols.values())
                                            keys = list(symbols.keys())
                                            position = vals.index(event_symbol_number)
                                            # get symbol of event in the tuple form.
                                            sym = keys[position]
                                            # the value the entity received in the last event recorded
                                            event_value = sym[1]
                                            # same value then extend the last event
                                            if event_value == v:
                                                new_finish = round((finish - start_time).total_seconds())
                                                # same start time and same symbol number only new finish time.
                                                new_event = (event[0], new_finish, event[2],)
                                                entities_events[entity][-1] = new_event
                                            else:
                                                # different value, add new event until the beginning of the next sliding window.
                                                sym_event = (entities[entity], v,)
                                                if symbols.get(sym_event, None) is None:
                                                    symbols[sym_event] = symbol_counter
                                                    symbol_counter += 1
                                                    symbols_counter_dict[(binning_methods_inv[b], k)] += 1
                                                # until the start of the next window.
                                                new_finish = round((finish - start_time).total_seconds())
                                                # create event
                                                last_event = (times[-1], new_finish, symbols[sym_event],)
                                                # add the event
                                                entities_events[entity].append(last_event)
                                        elif len(values) == 1:
                                            # len(values) = 1, only 1 value was received for the entity.
                                            # create new event for the duration of the entire window and add.
                                            new_finish = round((finish - start_time).total_seconds())
                                            sym_event = (entities[entity], values[0],)
                                            if symbols.get(sym_event, None) is None:
                                                symbols[sym_event] = symbol_counter
                                                symbol_counter += 1
                                                symbols_counter_dict[(binning_methods_inv[b], k)] += 1
                                            event = (times[0], new_finish, symbols[sym_event],)
                                            entities_events[entity].append(event)
                                    elif consider_last and i + w == len(df):
                                        # more than 1 value: compare last 2 and extend events or add event accordingly
                                        # 1 value: make event for the whole window.
                                        v = values[len(values) - 1]
                                        finish_time = round((window.iloc[-1]['time'] - start_time).total_seconds())
                                        if len(values) > 1:
                                            # they are different, add new event for the last value.
                                            # there have been more than 1 value changes.
                                            # last event
                                            event = entities_events[entity][-1]
                                            event_symbol_number = event[2]
                                            vals = list(symbols.values())
                                            keys = list(symbols.keys())
                                            position = vals.index(event_symbol_number)
                                            # get symbol of event in the tuple form.
                                            sym = keys[position]
                                            # the value the entity received in the last event recorded
                                            event_value = sym[1]
                                            if event_value == v:
                                                # same start time and same symbol number only new finish time.
                                                new_event = (event[0], finish_time, event[2],)
                                                entities_events[entity][-1] = new_event
                                            else:
                                                event_start = times[-1]
                                                event_finish = finish_time
                                                sym_event = (entities[entity], v,)
                                                if symbols.get(sym_event, None) is None:
                                                    symbols[sym_event] = symbol_counter
                                                    symbol_counter += 1
                                                    symbols_counter_dict[(binning_methods_inv[b], k)] += 1
                                                new_event = (event_start, event_finish, symbols[sym_event],)
                                                entities_events[entity].append(new_event)
                                        elif len(values) == 1:
                                            # a single value was received, add an event for the duration of the entire window.
                                            sym_event = (entities[entity], values[0],)
                                            if symbols.get(sym_event, None) is None:
                                                symbols[sym_event] = symbol_counter
                                                symbol_counter += 1
                                                symbols_counter_dict[(binning_methods_inv[b], k)] += 1
                                            entities_events[entity].append((times[0], finish_time, symbols[sym_event],))
        for (b, k) in itertools.product(binning_methods.values(), numbers_of_bins):
            entities_events = entities_events_dict[(binning_methods_inv[b], k)]
            sw_events = sw_events_dict[(binning_methods_inv[b], k)]
            entities = entities_dict[(binning_methods_inv[b], k)]
            # the key has to be the entity id and not the tuple of (PLC IP, register number). make the switch here.
            converted_events = {entities[entity]: entities_events[entity] for entity in entities_events.keys()}
            sw_events[i] = converted_events
    return sw_events_dict, symbols_dict, entities_dict


def make_input(pkt_df, binning_methods, numbers_of_bins, w, regs_to_use, consider_last=True, test_path=None,
               ready_symbols=None,
               ready_entities=None):
    if ready_symbols is None:
        ready_symbols = dict()
        ready_entities = dict()
    binning = {k_means_binning: 'kmeans', equal_frequency_discretization: 'equal_frequency',
               equal_width_discretization: 'equal_width'}
    binning_methods_inv = {k_means_binning: 'KMeans', equal_frequency_discretization: 'EqualFreq',
                           equal_width_discretization: 'EqualWidth'}
    # get a dictionary mapping from sw_number to the events in it.
    sw_events_dict, symbols_dict, entities_dict = define_events_in_sliding_windows(pkt_df, binning_methods,
                                                                                   numbers_of_bins, w, regs_to_use,
                                                                                   consider_last,
                                                                                   ready_symbols=ready_symbols,
                                                                                   ready_entites=ready_entities)
    for b in binning_methods.values():
        for k in numbers_of_bins:
            base_path = data.datasets_path + '//KL' + '//' + binning[b] + '_{}_{}'.format(k, w)
            if test_path is not None:
                base_path = test_path + '//' + binning[b] + '_{}_{}'.format(k, w)
            if not os.path.exists(base_path):
                Path(base_path).mkdir(parents=True, exist_ok=True)
            sw_events = sw_events_dict[(binning_methods_inv[b], k)]

            # keys are the window number.
            entity_index = 0
            entity_to_idx = {}
            for sw_num in sorted(sw_events.keys()):
                # hold the events of all the entities in that window.
                window_events = sw_events[sw_num]
                writeable = {}
                entity_counter = 0
                # counter number of entities which had some events. if there are none then there is nothing to
                # run the algorithm on so just pass on to the next window.
                for entity_id in sorted(window_events.keys()):
                    entity_events = window_events[entity_id]
                    if len(entity_events) > 0:
                        entity_counter += 1
                        writeable[entity_id] = entity_events
                if entity_counter == 0:
                    continue
                else:
                    window_path = base_path + '//#window_{}.csv'.format(sw_num)
                    with open(window_path, 'w', newline='') as window_file:
                        # now write the events of each entity to the file.
                        writer = csv.writer(window_file)
                        writer.writerow(['startToncepts'])
                        writer.writerow(['numberOfEntities,{}'.format(entity_counter)])
                        for writeable_entity in writeable.keys():
                            events_to_write = writeable[writeable_entity]
                            if entity_to_idx.get(writeable_entity, -1) == -1:
                                entity_to_idx[writeable_entity] = entity_index
                                entity_index += 1
                            writer.writerow(
                                ['{},{};'.format(str(writeable_entity), str(entity_to_idx[writeable_entity]))])
                            events_row = []
                            for event in events_to_write:
                                start = event[0]
                                finish = event[1]
                                symbol_number = event[2]
                                events_row += ['{},{},{};'.format(start, finish, symbol_number)]
                            writer.writerow(events_row)

    if test_path is None:
        # this means we are not testing. so, we are training and we need to save the symbols to avoid a redefinition of them
        # when testing and discovering.
        if not os.path.exists(KL_symbols):
            Path(KL_symbols).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(KL_entities):
            Path(KL_entities).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(KL_events):
            Path(KL_events).mkdir(exist_ok=True, parents=True)

        for b in binning_methods.values():
            for k in numbers_of_bins:
                suffix = '//{}_{}_{}'.format(binning[b], k, w)
                path_sym = KL_symbols + suffix

                sw_events = sw_events_dict[(binning_methods_inv[b], k)]
                entities = entities_dict[(binning_methods_inv[b], k)]
                symbols = symbols_dict[(binning_methods_inv[b], k)]
                with open(path_sym, mode='wb') as symbols_path:
                    pickle.dump(symbols, symbols_path)

                path_ent = KL_entities + suffix
                with open(path_ent, mode='wb') as entities_path:
                    pickle.dump(entities, entities_path)

                path_events = KL_events + suffix
                with open(path_events, mode='wb') as events_p:
                    pickle.dump(sw_events, events_p)


# split the raw data set into train and test. train classifier on train set.
# when injecting anomalies change the times in the raw data and then make input for the classifier.
# important to keep track of the malicious packets' indices.
def load_events_in_sliding_windows(b, k, w):
    suffix = '//{}_{}_{}'.format(b, k, w)
    path_sym = KL_symbols + suffix
    with open(path_sym, mode='rb') as symbols_path:
        symbols = pickle.load(symbols_path)
    path_ent = KL_entities + suffix
    with open(path_ent, mode='rb') as entities_path:
        entities = pickle.load(entities_path)
    path_events = KL_events + suffix
    with open(path_events, mode='rb') as events_p:
        events = pickle.load(events_p)
    return events, symbols, entities
