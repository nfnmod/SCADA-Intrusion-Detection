import statistics

import numpy as np

import data


class Transition:
    def __init__(self, initial_state, final_state, difference, id):
        self.initial_state = initial_state
        self.final_state = final_state
        self.difference = difference
        self.lower_limit = 0
        self.upper_limit = np.Infinity
        self.id = id


class Automaton:
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions


# get the registers that have changed in the transition.
def state_difference(initial_state, final_state):
    registers = initial_state.keys()
    change = dict()
    for register in registers:
        if initial_state[register] != final_state[register]:
            change[register] = final_state[register]
    return change


# get the number of different registers value between 2 states
def state_number_of_diffs(s1, s2):
    diff_counter = 0
    for reg in s1.keys():
        if s1[reg] != s2[reg]:
            diff_counter += 1
    return diff_counter


def new_state(state, states, registers):
    # assume it's different
    different = True
    for old_state in states:
        # assume the registers values are the same in the compared state
        same = True
        for reg in registers:
            # same is true only if all the registers values were identical
            same = same and (old_state[reg] == state[reg])
        # if we get same == True for some state is states this means that this is not a new state
        different = different and not same
    return different


def new_transition(transition, transitions):
    different = True
    for old_transition in transitions:
        old_initial = old_transition.initial_state
        old_final = old_transition.final_state
        t_initial = transition.initial_state
        t_final = transition.final_state
        # compare the states defining the transition
        same = state_number_of_diffs(old_initial, t_initial) == 0 and state_number_of_diffs(old_final, t_final) == 0
        # same logic as for states
        different = different and not same
    return different


def find_same(transition, transitions):
    for t in transitions:
        t_init = t.initial_state
        t_final = t.final_state
        transition_init = transition.initial_state
        transition_final = transition.final_state
        same = state_number_of_diffs(t_init, transition_init) == 0 and state_number_of_diffs(t_final,
                                                                                             transition_final) == 0
        if same:
            return t
    return None


# The registers don't always get their values together.
def make_automaton(registers, processed):
    states = []
    transitions = []
    transitions_times = []
    transition_id = 0

    for i in range(len(processed) - 1):
        # this saves a state and the time we were in the state.
        # the next packet is the next state.
        # so a pair of packets describes a transition and the time in state of the first packet is the duration.
        curr_pkt = processed.iloc[i]
        switched_state_pkt = processed.iloc[i + 1]
        state_duration = curr_pkt['time_in_state']
        # the starting state
        state = dict()
        # the state we switch into
        switched_state = dict()
        for reg in registers:
            # get the registers value from the state describing packet
            state[reg] = curr_pkt[str(reg)]
            switched_state[reg] = switched_state_pkt[str(reg)]
        # we now have the state, check if it's a new one
        if new_state(state, states, registers):
            states.append(state)
        if new_state(switched_state, states, registers):
            states.append(switched_state)
        # now we build the transition
        change = state_difference(state, switched_state)
        transition = Transition(state, switched_state, change, -1)
        # check if it's a new one
        if new_transition(transition, transitions):
            transition.id = transition_id
            transitions.append(transition)
            transitions_times.append([state_duration])
            transition_id += 1
        else:
            # look for the transition with the same states
            same_transition = find_same(transition, transitions)
            t_id = same_transition.id
            transitions_times[t_id].append(state_duration)
    # now we have built all the transitions and states. we can now make time constraints
    for transition in transitions:
        id = transition.id
        times = transitions_times[id]
        if len(times) == 1:
            mean_time_in_state = times[0]
            std_time_in_state = 0
        else:
            mean_time_in_state = statistics.mean(times)
            std_time_in_state = statistics.stdev(times)
        transition.upper_limit = mean_time_in_state + 3 * std_time_in_state
        transition.lower_limit = mean_time_in_state - 3 * std_time_in_state
    return Automaton(states, transitions)


# detect anomalies using the automaton: unknown state/transition or violation of time constrains
# for single PLC, registers holds the PLC-state defining registers.
# for multipe PLCs, registers holds a mapping from PLC to PLC-state defining registers.
def detect(automaton, processed, registers):
    decisions = []
    for i in range(len(processed) - 1):
        curr_pkt = processed.iloc[i]
        switched_state_pkt = processed.iloc[i + 1]
        state_duration = curr_pkt['time_in_state']
        # the starting state
        state = dict()
        # the state we switch into
        switched_state = dict()
        for reg in registers:
            # get the registers value from the state describing packet
            state[reg] = curr_pkt[str(reg)]
            switched_state[reg] = switched_state_pkt[str(reg)]
        if new_state(state, automaton.states, registers) or new_state(switched_state, automaton.states, registers):
            # unknown states, anomaly detected
            decisions.append(1)
        else:
            change = state_difference(state, switched_state)
            transition = Transition(state, switched_state, change, -1)
            if new_transition(transition, automaton.transitions):
                # unknown transition, anomaly detected
                decisions.append(1)
            else:
                same_transition = find_same(transition, automaton.transitions)
                upper = same_transition.upper_limit
                lower = same_transition.lower_limit
                if state_duration > upper or state_duration < lower:
                    # bad state duration, anomaly detected
                    decisions.append(1)
                else:
                    # everything is ok, no anomaly
                    decisions.append(0)
    return decisions
