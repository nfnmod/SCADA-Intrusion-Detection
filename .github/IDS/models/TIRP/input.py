# this class is responsible for preparing the input for the KarmaLego algorithm.
import csv
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


# compute the events in the sliding windows.
def define_events_in_sliding_windows(df, b, k, w, regs_to_use, consider_last=True, bin=True, ready_symbols=None,
                                     ready_entites=None):
    """

    :param ready_entites: same as for ready symbols
    :param ready_symbols: when testing, we get the symbols already discovered in the data set to ensure all the symbols
                          which can be found in the test data set are already defined.
    :param bin: bin or not (used for testing)
    :param regs_to_use: registers to consider
    :param df:
    :param b: binning method.
    :param k: number of bins.
    :param w: window size.
    :param consider_last: make the last event last until the start of the next window.
    :return:
    """
    if ready_symbols == dict():
        df = process_payload(df, regs_to_use)
        entities = define_entities(df)
    else:
        entities = ready_entites
    sw_events = {sw_num: [] for sw_num in range(len(df) - w)}
    symbols = ready_symbols
    symbol_counter = len(symbols.keys())
    # per window
    prev_entities = []

    for i in range(len(df) - w + 1):
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
        entities_events = {e: [] for e in window_entities}

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
                        reg_num = entity[1]
                        values = [float(payload[str(reg_num)])]
                        times = [round((pkt['time'] - start_time).total_seconds())]
                        for k_w in range(j + 1, w):
                            w_pkt = window.iloc[k_w]
                            # check for the entity in the payload.
                            if w_pkt['src_port'] == data.plc_port:
                                w_IP = w_pkt['src_ip']
                                w_reg_val = w_pkt['payload'].get(str(reg_num), None)
                                if w_IP == IP and w_reg_val is not None and (
                                        len(values) == 0 or values[-1] != w_reg_val):
                                    values.append(float(w_reg_val))
                                    times.append(round((w_pkt['time'] - start_time).total_seconds()))
                        # mark as checked.
                        checked_entities.append(entity)
                        # bin using all options to avoid reprocessing

                        if bin:
                            values = b(values, k)  # bin.
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
                                        new_event = (event_start, event_finish, symbols[sym_event],)
                                        entities_events[entity].append(new_event)
                                elif len(values) == 1:
                                    # a single value was received, add an event for the duration of the entire window.
                                    sym_event = (entities[entity], values[0],)
                                    if symbols.get(sym_event, None) is None:
                                        symbols[sym_event] = symbol_counter
                                        symbol_counter += 1
                                    entities_events[entity].append((times[0], finish_time, symbols[sym_event],))

        # the key has to be the entity id and not the tuple of (PLC IP, register number). make the switch here.
        converted_events = {entities[entity]: entities_events[entity] for entity in entities_events.keys()}
        sw_events[i] = converted_events
    return sw_events, symbols, entities


def make_input(pkt_df, b, k, w, regs_to_use, consider_last=True, test_path=None, ready_symbols=None,
               ready_entities=None):
    if ready_symbols is None:
        ready_symbols = dict()
        ready_entities = dict()
    binning = {k_means_binning: 'kmeans', equal_frequency_discretization: 'equal_frequency',
               equal_width_discretization: 'equal_width'}
    # get a dictionary mapping from sw_number to the events in it.
    sw_events, symbols, entities = define_events_in_sliding_windows(pkt_df, b, k, w, regs_to_use, consider_last,
                                                                    ready_symbols=ready_symbols,
                                                                    ready_entites=ready_entities)
    base_path = data.datasets_path + '\\KL' + '\\' + binning[b] + '_bins_{}_window_{}'.format(k, w)
    if test_path is not None:
        base_path = test_path
    if not os.path.exists(base_path):
        Path(base_path).mkdir(parents=True, exist_ok=True)

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
            window_path = base_path + '\\#window_{}.csv'.format(sw_num)
            with open(window_path, 'w', newline='') as window_file:
                # now write the events of each entity to the file.
                writer = csv.writer(window_file)
                writer.writerow(['startToncepts'])
                writer.writerow(['numberOfEntities,', entity_counter])
                for writeable_entity in writeable.keys():
                    events_to_write = writeable[writeable_entity]
                    if entity_to_idx.get(writeable_entity, -1) == -1:
                        entity_to_idx[writeable_entity] = entity_index
                        entity_index += 1
                    writer.writerow(['{},{};'.format(str(writeable_entity), str(entity_to_idx[writeable_entity]))])
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

        suffix = '\\{}_{}_{}'.format(binning[b], k, w)
        path_sym = KL_symbols + suffix

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
    suffix = '\\{}_{}_{}'.format(b, k, w)
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

