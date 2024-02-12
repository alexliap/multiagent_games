import itertools

def get_state_mapping(environment_size: tuple, possible_tile_states: tuple):
    state_row_dict={}
    i=0
    for comb in itertools.product(possible_tile_states, repeat=environment_size[0]*environment_size[1]):
        state_row_dict[str(comb)]=i
        i+=1
        
    return state_row_dict
