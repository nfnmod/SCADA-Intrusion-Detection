# this class is responsible for preparing the input for the KarmaLego algorithm.
import csv
import itertools

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

import data


# TODO: parameters value ranges for input preparation- w, k.
# TODO: random forest classifier- feature extraction from KarmaLego output and training.
# TODO: write a function that will calculate the changing registers of each PLC, consider them only, effects define_entities .
# TODO: test.

# ---------------------------------------------------------------------------------------------------------------------------#
# helper functions to bin data
def k_means_binning(values, n_bins):
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans').fit(values)
    labeled_data = k_means.transform(values)
    return labeled_data


def equal_width_discretization(values, n_bins):
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform').fit(values)
    labeled_data = k_means.transform(values)
    return labeled_data


def equal_frequency_discretization(values, n_bins):
    k_means = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit(values)
    labeled_data = k_means.transform(values)
    return labeled_data


# ---------------------------------------------------------------------------------------------------------------------------#
def define_entities(data_path, df_data=None, load_data=True):
    """

    :param data_path: read data from that path.
    :param df_data: the function may be called after loading data, holds the data itself.
    :param load_data: control the loading of data.
    :return:
    """
    if load_data:
        df = data.load(data.datasets_path, data_path)
    else:
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
                entity = (IP, register)  # the entity id of the register of the plc.
                existing_entity = entities.get(entity, None)  # check if it got an id already.
                if existing_entity is None:
                    entities[entity] = entity_counter  # set the id
                    entity_counter += 1  # update the id
    return entities


def get_pkt_entities(pkt):
    """

    :param pkt: a response packet
    :return: the entites in the packet.
    """
    IP = pkt['src_ip']
    payload = pkt['payload']
    # entities in the packet.
    packet_entities = [(IP, int(reg)) for reg in payload.keys()]
    return packet_entities


# compute the events in the sliding windows.
def define_events_in_sliding_windows(data_path, b, k, w, consider_last=True):
    """

    :param data_path: path to data used for TIRPs.
    :param b: binning method.
    :param k: number of bins.
    :param w: window size.
    :param consider_last: make the last event last until the start of the next window.
    :return:
    """
    df = data.load(data.datasets_path, data_path)
    start_time = (df.iloc[0])['time']
    entities = define_entities(data_path, df, False)
    sw_events = {sw_num: [] for sw_num in range(len(df) - w)}
    symbols = {}
    symbol_counter = 0

    for i in range(len(df) - w):
        checked_entities = []
        window = df.iloc[i: i + w]
        window_entities = []
        for w_i in range(len(window)):
            w_pkt = window[w_i]
            src = w_pkt['src_port']
            if src == data.plc_port:
                np.concatenate((window_entities, get_pkt_entities(w_pkt)))
        entities_events = {e: [] for e in window_entities}
        for j in range(w):
            pkt = window.iloc[i]
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
                        times = [round((pkt['time'] - start_time).total_seconds())]
                        for k in range(j + 1, w):
                            w_pkt = window.iloc[k]
                            # check for the entity in the payload.
                            if w_pkt['src_port'] == data.plc_port:
                                w_IP = w_pkt['src_ip']
                                w_reg_val = float(w_pkt['payload'].get(str(reg_num), None))
                                if w_IP == IP and w_reg_val is not None:
                                    values.append(w_reg_val)
                                    times.append(round((w_pkt['time'] - start_time).total_seconds() * 1000))
                        # mark as checked.
                        checked_entities.append(entity)
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
                                symbol = (entities[entity], value)
                                if symbols.get(symbol, None) is None:
                                    symbols[symbol] = symbol_counter
                                    symbol_counter += 1
                                event = (s_t, f_t, symbols[symbol])
                                entities_events[entity].append(event)
                            if consider_last:
                                # the last value-time pair. the symbol "holds" until the first packet of the next
                                v = values[len(values) - 1]
                                finish = df.iloc[i + w + 1]['time']
                                if n > 1:
                                    event = entities_events[entity][-1]
                                    event_value = event[2][1]
                                    if event_value == v:
                                        new_finish = round((finish - start_time).total_seconds() * 1000)
                                        new_event = (event[0], new_finish, event[2])
                                        entities_events[entity][-1] = new_event
                                    # if it's a new value then the event will be recorded when the window slides one step further.
                                else:
                                    # n = 1
                                    new_finish = round((finish - start_time).total_seconds() * 1000)
                                    event = (times[0], new_finish, (entities[entity], values[0]))
                                    entities_events[entity].append(event)
        # the key has to be the entity id and not the tuple of (PLC IP, register number). make the switch here.
        converted_events = {entities[entity]: entities_events[entity] for entity in entities_events.keys()}
        sw_events[i] = converted_events
    return sw_events


def make_input(data_path, b, k, w, consider_last=True):
    binning = {k_means_binning: 'kmeans', equal_frequency_discretization: 'equal_frequency',
               equal_width_discretization: 'equal_width'}
    # get a dictionary mapping from sw_number to the events in it.
    sw_events = define_events_in_sliding_windows(data_path, b, k, w, consider_last)
    base_path = data.datasets_path + '\\KL' + '\\' + binning[b] + '_bins_{}_window_{}'.format(k, w)
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
            window_path = base_path + '_#window_{}.csv'.format(sw_num)
            entity_index = 0
            with open(window_path, 'w', newline='') as window_file:
                # now write the events of each entity to the file.
                writer = csv.writer(window_file)
                writer.writerow('startToncepts')
                writer.writerow('numberOfEntities,{}'.format(entity_counter))
                for writeable_entity in writeable.keys():
                    events_to_write = writeable[writeable_entity]
                    writer.writerow('{},{}'.format(writeable_entity, entity_index))
                    entity_index += 1
                    events_row = ''
                    for event in events_to_write:
                        start = event[0]
                        finish = event[1]
                        symbol_number = event[2]
                        events_row += '{},{},{};'.format(start, finish, symbol_number)
                    writer.writerow(events_row)


def grid_input_preparation(data_path):
    binning_methods = [k_means_binning, equal_frequency_discretization, equal_width_discretization]
    number_of_bins = range(5, 11)
    windows = range(50, 225, 25)
    bins_window_options = itertools.product(number_of_bins, windows)
    options = itertools.product(binning_methods, bins_window_options)
    for option in options:
        b = option[0]
        k = option[1][0]
        w = option[1][1]
        make_input(data_path, b, k, w, consider_last=True)
