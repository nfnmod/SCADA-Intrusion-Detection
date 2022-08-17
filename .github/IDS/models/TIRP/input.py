# this class is responsible for preparing the input for the KarmaLego algorithm.
import data


# TODO: add functions for binning.
# TODO: see if the time needs an upscale.
# TODO: test.
# TODO: check about parameters sizes for input preparation- w, k.
# TODO: check about the classifiers used.

def define_entities(data_path, df_data=None, load_data=True):
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


# compute the events.
def define_events(data_path, b, k, w, consider_last=False):
    df = data.load(data.datasets_path, data_path)
    start_time = (df.iloc[0])['time']
    entities = define_entities(data_path, df, False)
    entities_events = {e: [] for e in entities}
    symbols = {}
    symbol_counter = 0

    for i in range(len(df) - w):
        checked_entities = []
        window = df.iloc[i, i + w]
        for j in range(w):
            pkt = window.iloc[i]
            # we want to look for entities in the packet.
            if pkt['src_port'] == data.plc_port:
                IP = pkt['src_ip']
                payload = pkt['payload']
                # entities in the packet.
                packet_entities = [(IP, int(reg)) for reg in payload.keys()]
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
                                    times.append(round((w_pkt['time'] - start_time).total_seconds()))
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
                                        new_finish = round((finish - start_time).total_seconds())
                                        new_event = (event[0], new_finish, event[2])
                                        entities_events[entity][-1] = new_event
                                    # if it's a new value then the event will be recorded when the window slides one step further.
                                else:
                                    # n = 1
                                    new_finish = round((finish - start_time).total_seconds())
                                    event = (times[0], new_finish, (entities[entity], values[0]))
                                    entities_events[entity].append(event)

                return entities_events
