"""
The purpose of this file is to provide classes and functions for parsing the output of KarmaLego.
It will be used for the featured definition of the classifier.
"""


def parse_raw_symbols(raw_symbols):
    """
    The format is s1-s2-s3-....-sn-. split around the - and return.
    """""
    return raw_symbols.split(sep='-')


def parse_raw_relations(raw_relations):
    """
    The format is r1.r2.r3....r(s * (s - 1) / 2).
    split around the dots and return.
    """
    return raw_relations.split(sep='.')


def parse_raw_instances(raw_instances):
    partial = raw_instances.split(sep=' ')
    entities = []
    events_dict = dict()
    for i in range(len(partial)):
        if i % 2 == 0:
            if events_dict.get(partial[i], None) is None:
                entities[partial[i]] = []
        else:
            start_finish = partial[i].split(sep='-')
            start = start_finish[0]
            finish = start_finish[1]
            event_times = (start, finish,)
            events_dict[partial[i - 1]].append(event_times)
    return events_dict


def parse_line(raw_line):
    pass


class TIRP:
    def __init__(self, ID, size, raw_symbols, raw_relations, raw_instances):
        self.ID = ID
        self.size = size
        self.symbols = parse_raw_symbols(raw_symbols)
        self.relations = parse_raw_relations(raw_relations)
        self.instances = parse_raw_instances(raw_instances)
