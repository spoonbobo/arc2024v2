from pprint import pprint

import numpy as np

from brain.cognitive_process import CognitiveProcess
from arc_dsl.dsl import *

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None

class Vision(CognitiveProcess):

    following = {
        'memory:*': None,
        'stimuli:vision': None
    }
    followers = {
        'memory:*': None
    }
    
    properties = {
        'shape': lambda x: x.shape, 
        'colors': lambda x: np.unique(x.flatten()),
        'objects': lambda x: objects(x, 
                                     univalued=True,
                                     diagonal=True, 
                                     without_bg=False),
        'frontiers': lambda x: frontiers(x),
        'partition': lambda x: partition(x)
    }

    attributes = {
        'objects': {
            'bbox': lambda x: bbox(x),
        },
        
        'frontiers': {
            'length': lambda x: len(x),
        }
    }
    
    def process_signal(self, signal, result):
        
        if not signal.pid == 'sensor':
            return

        problem = signal.data
        grids = {pair: {'inp': {}, 'out': {}} 
                    for pair in range(len(problem))}

        props = {pair: {'inp': {}, 'out': {}} 
                    for pair in range(len(problem))}
        
        attrs = {pair: {'inp': {}, 'out': {}} 
                    for pair in range(len(problem))}
        
        for idx, (inp, out) in enumerate(zip(problem.train_inputs,
                                             problem.train_outputs)):

            I = np.array(inp)
            O = np.array(out)
            
            grids[idx]['inp'] = I
            grids[idx]['out'] = O
            
            for prop, func in self.properties.items():
                props[idx]['inp'][prop] = safe_execute(func, I)
                props[idx]['out'][prop] = safe_execute(func, O)
                
            for attr, attr_funcs in self.attributes.items():
                I_objs = props[idx]['inp'].get(attr, [])
                O_objs = props[idx]['out'].get(attr, [])
                if attr not in attrs[idx]['inp']:
                    attrs[idx]['inp'][attr] = {}
                
                if attr not in attrs[idx]['out']:
                    attrs[idx]['out'][attr] = {}

                if I_objs is not None:
                    for objIdx, obj in enumerate(I_objs):
                        attrs[idx]['inp'][attr][objIdx] = {
                            func_name: safe_execute(func, obj)
                            for func_name, func in attr_funcs.items()
                        }

                if O_objs is not None:
                    for objIdx, obj in enumerate(O_objs):
                        attrs[idx]['out'][attr][objIdx] = {
                            func_name: safe_execute(func, obj)
                            for func_name, func in attr_funcs.items()
                        }

        result.data = {'problem_id': problem.problem_id,
                       'grids': grids, 
                       'props': props, 
                       'attrs': attrs}

def bbox(objects):
    coords = [(x, y) for _, (x, y) in objects]
    min_x = min(coords, key=lambda t: t[0])[0]
    max_x = max(coords, key=lambda t: t[0])[0]
    min_y = min(coords, key=lambda t: t[1])[1]
    max_y = max(coords, key=lambda t: t[1])[1]
    x1y1x2y2 = np.array([min_x, min_y, max_x, max_y])
    return x1y1x2y2
