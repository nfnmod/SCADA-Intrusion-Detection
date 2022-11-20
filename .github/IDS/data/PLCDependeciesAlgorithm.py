import statistics

import numpy as np
import pandas as pd


def find_frequent_transitions_sequences(packets, time_window, support):
    """

    :param packets: The processed data describing MODBUS traffic. The data should be of devices of one group and the registers values should
           be binned. (use data processing version 3.1 to get this. Consider only registers which change over time.)
    :param time_window: The time window to consider when trying to find a frequent subsequent state.
    :param support: Minimal support for a sequence/state to be considered frequent.
    :return: frequent state transitions which the PLCs in the group show.
    """

    """
    steps:
        1. find frequent states.
        2. for each frequent state s1:
            for each other frequent state s2:
                if s1 can't entail s2:
                    pass
                else:
                    occurrences = 0
                    for each appearance s1' of s1:
                        last = 0
                        search for the earliest appearance s2' (since the last one used) of s2 such that time(s2')-time(s1')<=t:
                            occurrences += 1
                            last = index of s2' in the appearances series
                    get support.
                    throw or keep. 
    """
    length = len(packets)
    # columns of registers.
    registers = list(packets.columns)[1:]
    # measure time.
    packets['timestamp'] = 0
    # appearances of states.
    states_appearances = {}

    for i in range(len(packets)):
        p = packets.iloc[i]

        # create a tuple of registers values describing the state.
        state_tuple = tuple()
        for reg in sorted(registers):
            state_tuple += ((reg, p[reg],),)

        appearances = states_appearances.get(state_tuple, [])
        appearances.append(i)
        states_appearances[state_tuple] = appearances

        # timestamp the current state.
        if i == 0:
            # first state is at time 0.
            packets.loc[0, 'timestamp'] = 0
        else:
            # any other state starts after the previous one has ended.
            packets.loc[i, 'timestamp'] = packets.loc[i - 1, 'timestamp'] + packets.loc[i - 1, 'time_in_state']

    frequent_states = []
    frequent_transitions = []
    # filter out the frequent states.
    for state, appearances in states_appearances:
        if len(appearances) / length >= support:
            frequent_states.append(state)

    # change into the form of (state,idx)
    flat_states = []
    for s in states_appearances.keys():
        idx_list = states_appearances[s]
        for state_idx in idx_list:
            flat_states.append((state_idx, s,))

    # sort by appearances, this is just like sorting by time of state occurrence.
    flat_states = sorted(flat_states, key=lambda idx_s: idx_s[0])
    # create all possible transitions.
    transitions_times = {}
    for i in range(len(flat_states) - 1):
        s = flat_states[i]
        f = flat_states[i + 1]
        s_idx = s[0]
        f_idx = f[0]
        time_delta = packets.loc[f_idx, 'timestamp'] - packets.loc[s_idx, 'timestamp']
        if time_delta <= time_window:
            transition = (s, f,)
            prev_times = transitions_times.get(transition, [])
            prev_times.append(time_delta)
            transitions_times[transition] = prev_times

    for t in sorted(transitions_times.keys(), key=lambda t: (t[0][0], t[1][0])):
        t_times = transitions_times[t]
        supp = len(t_times) / length
        if supp >= support:
            mean = statistics.mean(t_times)
            std = statistics.stdev(t_times)
            frequent_transitions.append(((t[0][1], t[1][1],), supp, mean, std))

    return frequent_transitions


def extract_features(frequent_transitions, registers):
    """

    :param frequent_transitions: output of the above algorithm.
    :param registers: the names of all the registers in the group which the transitions occur in.
    :return: a dataframe with features describing the frequent transitions.
    """
    entailing = ['entailing_{}'.format(reg) for reg in registers]
    following = ['following_{}'.format(reg) for reg in registers]
    delta_regs = ['changed_{}'.format(reg) for reg in registers]
    stats = ['support', 'mean_space', 'std_space']
    cols = np.concatenate([entailing, following, delta_regs, stats])
    extracted_df = pd.DataFrame(columns=cols)

    for frequent_transition in frequent_transitions:
        states = frequent_transition[0]
        # from.
        entailing_state = states[0]
        # to.
        following_state = states[1]
        # delta.
        entailing_regs = []
        for i in range(len(registers)):
            if entailing_state[i][1] != following_state[i][1]:
                entailing_regs.append(registers[i])

        support = frequent_transition[1]
        mean_space = frequent_transition[2]
        std_space = frequent_transition[3]

        frequent_transition_entry = {col: None for col in cols}
        for entailing_reg in entailing:
            reg_num = entailing_reg.split(sep='_')[1]
            frequent_transition_entry[entailing_reg] = entailing_state[reg_num[1]]

        for following_reg in following:
            reg_num = following_reg.split(sep='_')[1]
            frequent_transition_entry[following_reg] = entailing_state[reg_num[1]]

        for entailing_reg in delta_regs:
            reg_num = entailing_reg.split(sep='_')[1]
            if reg_num in entailing_regs:
                frequent_transition_entry[entailing_reg] = 1
            else:
                frequent_transition_entry[entailing_reg] = 0

        frequent_transition_entry['support'] = support
        frequent_transition_entry['mean_space'] = mean_space
        frequent_transition_entry['std_space'] = std_space
        transition_df = pd.DataFrame.from_dict(data={'0': frequent_transition_entry}, orient='index', columns=cols)
        extracted_df = pd.concat([extracted_df, transition_df], ignore_index=True)

    return extracted_df
