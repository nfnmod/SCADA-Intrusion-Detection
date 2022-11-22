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

    # used for ordering transition sequences.
    def comparator(t_seq_and_idx):
        indices = t_seq_and_idx[0]
        my_length = len(indices)
        if my_length < longest:
            return indices + (0,) * (longest - my_length)
        return indices

    length = len(packets)
    # columns of registers.
    registers = list(packets.columns)[2:]
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
    for s in frequent_states:
        idx_list = states_appearances[s]
        for state_idx in idx_list:
            flat_states.append((state_idx, s,))

    # sort by appearances, this is just like sorting by time of state occurrence.
    flat_states = sorted(flat_states, key=lambda idx_s: idx_s[0])
    # create all possible transitions.
    transitions_times = {}
    transitions_indices = {}
    for i in range(len(flat_states) - 1):
        s = flat_states[i]
        f = flat_states[i + 1]
        s_idx = s[0]
        starting_state = s[1]
        f_idx = f[0]
        finished_state = f[1]
        time_delta = packets.loc[f_idx, 'timestamp'] - packets.loc[s_idx, 'timestamp']
        if time_delta <= time_window:
            transition = (starting_state, finished_state,)

            # add times
            prev_times = transitions_times.get(transition, [])
            prev_times.append(time_delta)
            transitions_times[transition] = prev_times

            # add indices
            prev_indices = transitions_indices.get(transition, [])
            prev_indices.append((s_idx, f_idx,))
            transitions_indices[transition] = prev_indices

    # discover frequent transitions of length 1.
    for transition in transitions_times.keys():
        transition_time_spaces = transitions_times[transition]
        t_supp = len(transition_time_spaces) / length  # number of occurrences of the transition.

        if t_supp >= support:
            frequent_transitions.append(transition)

    # now we want to extend the transitions into longer sequences which are frequent
    # and also occur within a time window of at most time_window time units.

    # longest sequence length. used for sorting.
    longest = 0

    fully_extended = []
    fully_extended_times = {}
    fully_extended_indices = {}

    prev_times = transitions_times
    prev_indices = transitions_indices

    # extend as long as possible.
    while len(frequent_transitions) > 1:
        flat_transitions = []
        times = {}
        indices = {}
        components = {}

        # change into form of (transitions states, indices) = ((s1->s2->..s_n), (idx1,idx2...idx_n))
        for frequent_transition in frequent_transitions:
            # number of transitioned states.
            transition_sequence_length = len(frequent_transition)
            longest = max(longest, transition_sequence_length)

            t_seq_idx = transitions_indices[frequent_transition]
            flat_t_seq = (t_seq_idx, frequent_transition,)
            flat_transitions.append(flat_t_seq)

        flat_transitions = sorted(flat_transitions, key=lambda x: comparator(x))
        # from the sorting function we know that the transitions appear in increasing order
        # of their occurrence time and that shorter sequences are favored by the comparator.
        for i in range(len(flat_transitions) - 1):
            start = flat_transitions[i]
            start_indices = start[0]
            start_states = start[1]

            last_state = start_states[-1]
            last_idx = start_indices[-1]

            first_state_idx = start_indices[0]
            start_time = packets.loc[first_state_idx, 'timestamp']

            # we need to check every sequence which occurred after start.
            # if we extended, then break, remove them both from the candidates for the current iteration.
            # save the extended one aside for the next iteration.
            # if not, save start aside, remove from the candidates for the next iterations and continue.
            for j in range(i + 1, len(flat_transitions)):
                possible_end = flat_transitions[j]
                end_indices = possible_end[0]
                end_states = possible_end[1]

                # we need start to end with the first state of possible_end
                first_state = end_states[0]
                first_idx = end_indices[0]

                last_state_idx = end_indices[-1]
                end_time = packets.loc[last_state_idx, 'timestamp']
                time_delta = end_time - start_time
                if first_state != last_state or first_idx != last_idx or time_delta > time_window:
                    continue
                else:
                    # we can append them.
                    # extended sequence.
                    start_part = start_states
                    end_part = end_states[1:]
                    extended_transition = start_part + end_part

                    # extended indices.
                    start_part_indices = start_indices
                    end_part_indices = end_indices[1:]
                    extended_transition_indices = start_part_indices + end_part_indices

                    # add times
                    known_times = times.get(extended_transition, [])
                    known_times.append(time_delta)
                    times[extended_transition] = known_times

                    # add indices
                    known_indices = indices.get(extended_transition, [])
                    known_indices.append(extended_transition_indices)
                    indices[extended_transition] = known_indices

                    # save the smaller sequences which were extended.
                    # this will be used in case the extension isn't frequent.
                    if components.get(extended_transition, None) is None:
                        components[extended_transition] = [start, end_part]

        keep = {t: False for t in frequent_transitions}

        for i in range(len(flat_transitions) - 1):
            # now we filter out the infrequent/fully extended sequences.
            indices_and_transition = flat_transitions[i]

            t_indices = indices_and_transition[0]
            transition = indices_and_transition[1]

            for j in range(i + 1, len(flat_transitions) - 1):
                following_transition_and_indices = flat_transitions[j]

                following_indices = following_transition_and_indices[0]
                following_transition = following_transition_and_indices[1]

                extension_states = transition + following_transition[1:]

                parts = components.get(extension_states, [])
                if not parts:
                    continue
                else:
                    # this is an actual extension.
                    time_spaces = times[extension_states]
                    occurrences = len(time_spaces)
                    t_supp = occurrences / length

                    if t_supp >= support:
                        keep[transition] = True
                        keep[following_transition] = True
                    # this will be performed when the transition wasn't extended by any other one.
                    if t_supp < support:
                        # not frequent.

                        if not parts:
                            # it means it was never extended.
                            fully_extended.append(transition)
                            fully_extended_times[transition] = prev_times[transition]
                            fully_extended_indices[transition] = prev_indices[transition]
                        else:
                            for part in parts:
                                fully_extended.append(part)
                                fully_extended_times[part] = prev_times[part]
                                fully_extended_indices[part] = prev_indices[part]

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
