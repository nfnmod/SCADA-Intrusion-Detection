import statistics

import numpy as np
import pandas as pd


def cant_entail(s1, s2, states_appearances):
    s1_appearances = states_appearances[s1]
    s2_appearances = states_appearances[s2]

    # idx of last appearance of s1
    first_s1_idx = s1_appearances[0]
    # idx of first appearance of s2.
    last_s2_idx = s2_appearances[-1]

    if first_s1_idx > last_s2_idx:
        return True
    return False


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

    num_frequent_states = len(frequent_states)
    for i in range(num_frequent_states):
        s1 = frequent_states[i]
        for j in range(num_frequent_states):
            s2 = frequent_states[j]
            if s1 == s2 or cant_entail(s1, s2, frequent_states):
                pass
            else:
                # for each occurrence of s1, find some occurrence of s2 such that
                # s1's occurrence has happened at most t time units before the one of s2.
                entailments = 0
                last_used = -1
                sorted_appearances_s1 = sorted(states_appearances[s1])
                sorted_appearances_s2 = sorted(states_appearances[s2])
                sorted_appearances_s2_length = len(sorted_appearances_s2)
                s1_s2_times = []

                for s1_occurrence_idx in sorted_appearances_s1:
                    s1_occurrence_time = packets.loc[s1_occurrence_idx, 'timestamp']
                    for appearance_idx in range(last_used + 1, sorted_appearances_s2_length):
                        s2_occurrence_idx = sorted_appearances_s2[appearance_idx]
                        if s2_occurrence_idx < s1_occurrence_idx:
                            pass
                        else:
                            # entail the closest one in time.
                            s2_occurrence_time = packets.loc[s2_occurrence_idx, 'timestamp']
                            difference = (s2_occurrence_time - s1_occurrence_time)
                            if 0 < difference <= time_window:
                                entailments += 1
                                last_used = appearance_idx
                                s1_s2_times.append((s1_occurrence_time, s2_occurrence_time,))
                                # break, let the next appearance of s1 entail some appearance of s2.
                                break

                # check if frequent.
                supp = entailments / length
                if supp >= support:
                    t = (s1, s2,)
                    time_spaces = []
                    for (s1_occurrence_time, s2_occurrence_time,) in s1_s2_times:
                        time_spaces.append(s2_occurrence_time - s1_occurrence_time)
                    mean_space = statistics.mean(time_spaces)
                    std_space = statistics.stdev(time_spaces)
                    frequent_transitions.append((t, supp, mean_space, std_space))

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
