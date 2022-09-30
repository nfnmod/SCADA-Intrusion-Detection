# this class is responsible for preparing the input for the KarmaLego algorithm.
import csv
import itertools

import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

import data

payload_col_number = 6


# ---------------------------------------------------------------------------------------------------------------------------#
# helper functions to bin data
def k_means_binning(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))


def equal_width_discretization(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))


def equal_frequency_discretization(values, n_bins):
    values = np.reshape(values, newshape=(-1, 1))
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit(values)
    labeled_data = k_means.transform(values)
    return np.reshape(labeled_data, newshape=(1, -1))


# ---------------------------------------------------------------------------------------------------------------------------#
# the function will leave in the payload only the registers which change their values
def process_payload(df, stats_dict):
    payloads = []
    for i in range(len(df)):
        pkt = df.iloc[i]
        src = pkt['src_port']
        new_payload = {}
        if src == data.plc_port:
            payload = pkt['payload']
            plc_ip = pkt['src_ip']
            regs_vals_stds = stats_dict[plc_ip]
            new_payload = {}
            for reg in payload.keys():
                reg_stats = regs_vals_stds.get(reg, None)
                if reg_stats is not None:
                    num_vals = reg_stats[0]
                    std = reg_stats[1]  # maybe use in the future.
                    # pick only the registers which have changing values.
                    if num_vals > 1:
                        new_payload[reg] = payload[reg]

        # done processing this payload, now switch the original and the new one.
        payloads.append(new_payload)
    df.loc[:, 'payload'] = payloads


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
            payload_registers = [int(reg) for reg in payload.keys()]
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
    for reg in payload.keys():
        packet_entities.add((IP, int(reg),))
    return packet_entities


# compute the events in the sliding windows.
def define_events_in_sliding_windows(df, b, k, w, stats_dict, consider_last=True, bin=True):
    """

    :param bin: bin or not (used for testing)
    :param stats_dict:
    :param df:
    :param b: binning method.
    :param k: number of bins.
    :param w: window size.
    :param consider_last: make the last event last until the start of the next window.
    :return:
    """
    start_time = (df.iloc[0])['time']
    process_payload(df, stats_dict)  # works in place.
    entities = define_entities(df)
    sw_events = {sw_num: [] for sw_num in range(len(df) - w)}
    symbols = dict()
    symbol_counter = 0

    for i in range(len(df) - w + 1):
        checked_entities = []
        window = df.iloc[i: i + w]
        window_entities = set()
        for w_i in range(len(window)):
            w_pkt = window.iloc[w_i]
            src = w_pkt['src_port']
            if src == data.plc_port:
                window_entities = window_entities.union(get_pkt_entities(w_pkt))
        entities_events = {e: [] for e in window_entities}
        for j in range(w):
            pkt = window.iloc[j]
            # we want to look for entities in the packet.
            if pkt['src_port'] == data.plc_port:
                IP = pkt['src_ip']
                payload = pkt['payload']
                # entities in the packet.
                packet_entities = get_pkt_entities(pkt)
                for entity in packet_entities:
                    # find the values the entity received in the window and bin them.
                    # find the times at which the entity has received the values.
                    # create events and add them to the list of the entity's events.
                    # mark the entity as checked.
                    # no need to consider previous packets because the entity doesn't appear in them. If it were to appear in them
                    # then it wouldn't be unchecked.
                    if entity not in checked_entities:
                        reg_num = entity[1]
                        values = [float(payload[str(reg_num)])]
                        times = [round((pkt['time'] - start_time).total_seconds() * 1000)]
                        for k_w in range(j + 1, w):
                            w_pkt = window.iloc[k_w]
                            # check for the entity in the payload.
                            if w_pkt['src_port'] == data.plc_port:
                                w_IP = w_pkt['src_ip']
                                w_reg_val = w_pkt['payload'].get(str(reg_num), None)
                                if w_IP == IP and w_reg_val is not None:
                                    values.append(float(w_reg_val))
                                    times.append(round((w_pkt['time'] - start_time).total_seconds() * 1000))
                        # mark as checked.
                        checked_entities.append(entity)
                        if bin:
                            values = b(values, k)  # bin.
                        n = 0
                        if len(values) > 0:
                            # all but last value-time pair.
                            while n < len(values) - 1:
                                value = values[n]
                                s_t = times[n]
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
                                    # len(values) > 1 means that a value was received at least 2 times.
                                    # there has to be at least 1 event because len(values) - 1 is at least 1 so the loop is executed and an event is added.
                                    # if 2 last are the same, extend the event.
                                    # otherwise, nothing. the event for that value will be recorded in the next window.
                                    event = entities_events[entity][-1]
                                    event_symbol_number = event[2]
                                    vals = list(symbols.values())
                                    keys = list(symbols.keys())
                                    position = vals.index(event_symbol_number)
                                    sym = keys[position]
                                    event_value = sym[1]
                                    if event_value == v:
                                        new_finish = round((finish - start_time).total_seconds() * 1000)
                                        new_event = (event[0], new_finish, event[2],)
                                        entities_events[entity][-1] = new_event
                                    # if it's a new value then the event will be recorded when the window slides one step further.
                                else:
                                    # len(values) = 1, only 1 value was received for the entity.
                                    # create new event for the duration of the entire window and add.
                                    new_finish = round((finish - start_time).total_seconds() * 1000)
                                    sym_event = (entities[entity], values[0],)
                                    if symbols.get(sym_event, None) is None:
                                        symbols[sym_event] = symbol_counter
                                        symbol_counter += 1
                                    event = (times[0], new_finish, symbols[sym_event],)
                                    entities_events[entity].append(event)
                            elif consider_last and i + w == len(df):
                                # if len(values) > 1, compare 2 last values received.
                                # there has to be an event for the entity for the reasons mentioned above.
                                # if they are the same, extend.
                                # otherwise, add an event.
                                v = values[len(values) - 1]
                                finish_time = round((window.iloc[-1]['time'] - start_time).total_seconds() * 1000)
                                if len(values) > 1:
                                    event = entities_events[entity][-1]
                                    event_symbol_number = event[2]
                                    vals = list(symbols.values())
                                    keys = list(symbols.keys())
                                    position = vals.index(event_symbol_number)
                                    sym = keys[position]
                                    event_value = sym[1]
                                    if event_value == v:
                                        # same value, extend last event to the end of the window.
                                        new_finish = finish_time
                                        new_start = event[0]
                                        new_event = (new_start, new_finish, event[2])
                                        entities_events[entity][-1] = new_event
                                    else:
                                        # they are different, add new event for the last value.
                                        event_start = times[-1]
                                        event_finish = finish_time
                                        sym_event = (entities[entity], v,)
                                        if symbols.get(sym_event, None) is None:
                                            symbols[sym_event] = symbol_counter
                                            symbol_counter += 1
                                        new_event = (event_start, event_finish, symbols[sym_event])
                                        entities_events[entity].append(new_event)
                                elif len(values) == 1:
                                    # a single value was received, add an event for the duration of the entire window.
                                    sym_event = (entities[entity], values[0],)
                                    if symbols.get(sym_event, None) is None:
                                        symbols[sym_event] = symbol_counter
                                        symbol_counter += 1
                                    entities_events[entity].append((times[0], finish_time, symbols[sym_event]))
        # the key has to be the entity id and not the tuple of (PLC IP, register number). make the switch here.
        converted_events = {entities[entity]: entities_events[entity] for entity in entities_events.keys()}
        sw_events[i] = converted_events
    return sw_events, symbols, entities


def make_input(pkt_df, b, k, w, stats_dict, consider_last=True, test_path=None):
    binning = {k_means_binning: 'kmeans', equal_frequency_discretization: 'equal_frequency',
               equal_width_discretization: 'equal_width'}
    # get a dictionary mapping from sw_number to the events in it.
    sw_events, symbols, entities = define_events_in_sliding_windows(pkt_df, b, k, w, stats_dict, consider_last)
    base_path = data.datasets_path + '\\KL' + '\\' + binning[b] + '_bins_{}_window_{}'.format(k, w)
    if test_path is not None:
        base_path = test_path
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
            entity_index = 0
            with open(window_path, 'w', newline='') as window_file:
                # now write the events of each entity to the file.
                writer = csv.writer(window_file)
                writer.writerow('startToncepts')
                writer.writerow('numberOfEntities,{}'.format(entity_counter))
                for writeable_entity in writeable.keys():
                    events_to_write = writeable[writeable_entity]
                    writer.writerow('{},{}'.format(entities[writeable_entity], entity_index))
                    entity_index += 1
                    events_row = ''
                    for event in events_to_write:
                        start = event[0]
                        finish = event[1]
                        symbol_number = event[2]
                        events_row += '{},{},{};'.format(start, finish, symbol_number)
                    writer.writerow(events_row)


# split the raw data set into train and test. train classifier on train set.
# when injecting anomalies change the times in the raw data and then make input for the classifier.
# important to keep track of the malicious packets' indices.
def discover(plc_df, b, k, w, consider_last, stats_dict, test_path=None):
    binning = {k_means_binning: 'kmeans', equal_frequency_discretization: 'equal_frequency',
               equal_width_discretization: 'equal_width'}
    base_path = data.datasets_path + '\\KL' + '\\whole_input\\all_' + binning[b] + '_bins_{}_window_{}.csv'.format(k, w)
    if test_path is not None:
        base_path = test_path
    # get a dictionary mapping from sw_number to the events in it.
    sw_events, symbols, entities = define_events_in_sliding_windows(plc_df, b, k, w, stats_dict, consider_last)
    entity_index = 0
    with open(base_path, 'w') as all_TIRPs:
        writer = csv.writer(all_TIRPs)
        writer.writerow('startToncepts')
        writer.writerow('numberOfEntities,{}'.format(len(entities.keys())))
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
                for writeable_entity in writeable.keys():
                    events_to_write = writeable[writeable_entity]
                    writer.writerow('{},{}'.format(entities[writeable_entity], entity_index))
                    entity_index += 1
                    events_row = ''
                    for event in events_to_write:
                        start = event[0]
                        finish = event[1]
                        symbol_number = event[2]
                        events_row += '{},{},{};'.format(start, finish, symbol_number)
                    writer.writerow(events_row)


def grid_input_preparation():
    binning_methods = [k_means_binning, equal_frequency_discretization, equal_width_discretization]
    number_of_bins = range(5, 11)
    windows = range(200, 325, 25)
    pkt_df = data.load(data.datasets_path, "modbus")
    IP = data.plc
    # consider only response packets from the PLC.
    plc_df = pkt_df.loc[(pkt_df['src_ip'] == IP) & (pkt_df['src_port'] == data.plc_port)]
    """indices = range(len(plc_df))
    train, test = train_test_split(indices, test_size=0.2, random_state=42)
    train_df = plc_df.iloc[train]
    test_df = plc_df.iloc[test]"""
    stats_dict = data.get_plcs_values_statistics(plc_df, 5, to_df=False)
    bins_window_options = itertools.product(number_of_bins, windows)
    options = itertools.product(binning_methods, bins_window_options)
    # data.dump(data.datasets_path, "train_raw_automaton_TIRP", train_df)
    # data.dump(data.datasets_path, "test_raw_automaton_TIRP", test_df)
    for option in options:
        b = option[0]
        k = option[1][0]
        w = option[1][1]
        discover(plc_df, b, k, w, consider_last=True, stats_dict=stats_dict)
        make_input(plc_df, b, k, w, stats_dict, consider_last=True)
        print("made input!")


if __name__ == '__main__':
    grid_input_preparation()
