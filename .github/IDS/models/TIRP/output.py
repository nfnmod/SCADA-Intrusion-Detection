"""
The purpose of this file is to provide classes and functions for parsing the output of KarmaLego.
It will be used for the featured definition of the classifier.
"""
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

ID = 0


class TIRP:
    def __init__(self, TIRP_ID, size, supporting_entities, instances_to_entities_relation, raw_symbols, raw_relations,
                 raw_instances):
        self.TIRP_ID = TIRP_ID
        self.size = size
        self.supporting_entities = supporting_entities
        self.instances_to_entities_relation = instances_to_entities_relation
        self.symbols = parse_raw_symbols(raw_symbols)
        self.relations = parse_raw_relations(raw_relations)
        self.instances = parse_raw_instances(raw_instances, size)
        self.mean_duration = calculate_mean_duration(self.instances)
        self.support = None


def parse_raw_symbols(raw_symbols):
    """
    The format is s1-s2-s3-....-sn-. split around the - and return.
    """""
    split = raw_symbols.split(sep='-')
    split = split[0: len(split) - 1]
    return split


def parse_raw_relations(raw_relations):
    """
    The format is r1.r2.r3....r(s * (s - 1) / 2).
    split around the dots and return.
    """
    split = raw_relations.split(sep='.')
    split = split[0: len(split) - 1]
    return split


def parse_raw_instances(raw_instances, size):
    partial = raw_instances.split(sep=' ')
    events_dict = dict()
    last_entity = None
    for i in range(len(partial)):
        curr_part = partial[i]
        if curr_part[0] == '[':
            # this part of the string represents the times of the instances of the TIRP for a specific entity.
            start_finish = partial[i].split(sep='-')
            # loop over instances.
            start = -1
            finish = -1
            for part_num in range(2 * size - 1):
                # each instance is split into a list of size 2 * |TIRP| + 1.
                # each part that starts with [ stands for the starting time of an event.
                # each part that starts with a number and has no ] in it stands for the finish time of one event and
                # the beginning of the next one.
                # each part that ends with a ] stands for the finish time of an event.
                timestamp = start_finish[part_num]
                if timestamp[0] == '[':
                    start = int(timestamp[1:])
                elif timestamp[-1] == ']':
                    finish = int(timestamp[:-1])
                    event_times = (start, finish,)
                    events_dict[last_entity].append(event_times)
                else:
                    finish_index = timestamp.index(']')
                    start_index = timestamp.index('[')
                    finish = int(timestamp[:finish_index])
                    event_times = (start, finish,)
                    events_dict[last_entity].append(event_times)
                    start = int(timestamp[start_index + 1:])
        else:
            # this part of the string represents an entity ID.
            last_entity = partial[i]
            if last_entity not in events_dict.keys():
                events_dict[last_entity] = []
    return events_dict


def parse_line(raw_line):
    parts = raw_line.split(sep=' ')
    size = int(parts[0])
    raw_symbols = parts[1]
    raw_relations = parts[2]
    supporting_entities = int(parts[3])
    instances_to_entities_relation = float(parts[4])
    raw_instances_split = parts[5:]
    raw_instances = raw_instances_split[0]
    for part in range(1, len(raw_instances_split)):
        raw_instances += (" " + raw_instances_split[part])
    global ID
    ID += 1
    return TIRP(ID, size, supporting_entities, instances_to_entities_relation, raw_symbols, raw_relations,
                raw_instances)


def compare_TIRPs(TIRP1: TIRP, TIRP2: TIRP) -> bool:
    if TIRP1.size != TIRP2.size:
        return False
    elif TIRP1.supporting_entities != TIRP2.supporting_entities:
        return False
    elif TIRP1.symbols != TIRP2.symbols:
        return False
    elif TIRP1.relations != TIRP2.relations:
        return False
    return True


def calculate_mean_duration(instances):
    instances_count = 0
    total_duration = 0
    for entity_id in instances.keys():
        for instance in instances[entity_id]:
            total_duration += (instance[1] - instance[0])
            instances_count += 1
    return total_duration / instances_count


def calculate_support(instances, TIRPs_count):
    instances_count = 0
    for entity_id in instances.keys():
        instances_count += len(instances[entity_id])
    return instances_count / TIRPs_count


def get_number_of_instances(instances):
    instances_counter = 0
    for entity_id in instances.keys():
        instances_counter += len(instances[entity_id])
    return instances_counter


def found_TIRP(TIRPs_in_window, tirp):
    """

    :param TIRPs_in_window: a list containing the tirps in some window.
    :param tirp: a tirp
    :return: the tirp in the window if it's in the window, None otherwise
    """
    for TIRP_in_window in TIRPs_in_window:
        if compare_TIRPs(TIRP_in_window, tirp):
            return TIRP_in_window
    return None


def parse_output(window_TIRPs_folder, tirps_path, train=True):
    """

    :param window_TIRPs_folder: the path to a folder which contains the output of KL on the separate windows.
    :return: a dataframe to be used for training classifiers.
    """
    # 1. iterate over the rows of "all TIRPs" file parse and assign an index to each one- the initial discovery step.
    # 2. define features according to the number of TIRPs: [tirp1, ..., tirp_n] + [tirp1_md, ...., tirp_n_md] + [tirp1_s, ..., tirp_n_s].
    # The first group are binary valued indicators for the presence of the TIRP in the window. The second and third groups are the
    # mean duration and support of the respective TIRPs, respectively.
    # 3. iterate over the window files, for each one mark the found TIRPs and the mean duration and support and create a dataframe entry.
    # 4. return the dataset.

    # 1
    # save the found tirps to be used when testing.
    if train:
        TIRPs = []
        for output_file in os.listdir(window_TIRPs_folder):
            with open(window_TIRPs_folder + '\\' + output_file, mode='r') as TIRPs_file:
                for TIRP_line in TIRPs_file:
                    TIRP = parse_line(TIRP_line)
                    if found_TIRP(TIRPs, TIRP) is None:
                        TIRPs.append(TIRP)
        with open(tirps_path, mode='wb') as tirps_file:
            pickle.dump(TIRPs, tirps_file)
    else:
        # testing, so need to load the TIRPs
        with open(tirps_path, mode='wb') as tirps_file:
            TIRPs = pickle.load(tirps_file)
    # 2
    TIRP_count = len(TIRPs)
    presence = [str(tirp.TIRP_ID) for tirp in TIRPs]
    mean_durations = ['{}_md'.format(tirp.TIRP_ID) for tirp in TIRPs]
    supports = ['{}_hs'.format(tirp.TIRP_ID) for tirp in TIRPs]
    features = np.concatenate((presence, mean_durations, supports))
    windows_TIRPs_df = pd.DataFrame(columns=features)

    # 3
    for window_file in os.listdir(window_TIRPs_folder):
        # get the TIRPs in the window
        # calculate number of tirp instances for the HS features.
        # check what tirps are found and create the new sample.
        window_TIRPs_count = 0
        TIRPs_in_window = []
        # define the TIRPs in the window.
        with open(window_file, mode='r') as window_TIRPs:
            for raw_TIRP in window_TIRPs:
                t = parse_line(raw_TIRP)
                TIRPs_in_window.append(t)
                window_TIRPs_count += get_number_of_instances(t.instances)
        # fill in the missing values for the horizontal support.
        for TIRP_in_window in TIRPs_in_window:
            TIRP_in_window.support = calculate_support(TIRP_in_window.instances, window_TIRPs_count)
        # create an entry for the dataframe.
        window_presence = np.zeros(TIRP_count)
        window_mean_durations = np.zeros(TIRP_count)
        window_horizontal_supports = np.zeros(TIRP_count)
        for tirp_idx in range(TIRP_count):
            tirp = TIRPs[tirp_idx]
            search_res = found_TIRP(TIRPs_in_window, tirp)
            if search_res is not None:
                window_presence[tirp_idx] = 1
                window_mean_durations[tirp_idx] = search_res.mean_duration
                window_horizontal_supports[tirp_idx] = search_res.support
        window_features = np.concatenate((window_presence, window_mean_durations, window_horizontal_supports))
        # add to the dataframe
        window_df = pd.DataFrame.from_dict(columns=features, data={'0': window_features}, orient='index')
        windows_TIRPs_df = pd.concat((windows_TIRPs_df, window_df), axis=0, ignore_index=True)

    # min-max scaling of the columns.
    for col in windows_TIRPs_df.columns:
        scaler = MinMaxScaler()
        np_col = windows_TIRPs_df[col].to_numpy().reshape(-1, 1)
        scaler.fit(np_col)
        windows_TIRPs_df[col] = scaler.transform(np_col)

    return windows_TIRPs_df
