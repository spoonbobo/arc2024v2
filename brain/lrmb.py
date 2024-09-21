import inspect
from multiprocessing.shared_memory import SharedMemory

from arc_dsl import dsl
from brain.cognitive_process import Signal
from brain.layers_def import CPs, Ls

class LRMB:

    def __init__(self):
        self.shm = {}
        self.global_mem = SharedMemory(create=True,
                                       size=64,
                                       name='global_mem')
        self.text_embedder = None # TODO
        self.primitives = None
        self.primitives_map = None

        self.grammar = self.intiialize_dsl()
        self.layers = self.initialize_layers()
        self.initialize_connections()
        
    def initialize_dsl(self):
        self.primitives = []
        for name, primitive in inspect.getmembers(dsl):
            if inspect.isfunction(primitive):
                self.primitives.append(self.parse_primitive(primitive, name))

        self.primitives_map = {idx: primitive
                               for idx, primitive in enumerate(self.primitives)}

    def initialize_layers(self):
        layers = {}
        for layer, layer_name in Ls.items():
            self.shm[layer_name] = SharedMemory(create=True,
                                           size=1024,
                                           name=f'{layer}_shm')
            
            layers[layer_name] = {name: process(pid=name, 
                                                layer=layer,
                                                primitives=self.primitives,
                                                primitives_map=self.primitives_map)
                             for name, process in CPs[layer_name].items()}
        return layers
    
    def initialize_connections(self):
        for layer in self.layers.values():
            for process in layer.values():
                for follower in process.followers:
                    if follower == '*':
                        for layer in self.layers.values():
                            for subscriber in layer.values():
                                process.subscribers[subscriber.pid] = subscriber.signal_queue
                        continue
                
                    L, CP = follower.split(':')
                    if CP == '*':
                        for subscriber in self.layers[L].values():
                            process.subscribers[subscriber.pid] = subscriber.signal_queue
                        continue
                    
                    process.subscribers[follower] = self.layers[L][CP].signal_queue

    def activate(self):
        for layer in self.layers.values():
            for process in layer.values():
                process.start()

    @staticmethod
    def parse_primitive(primitive, name):
        return {
            'name': name,
            'function': primitive,
            'arg_types': [arg_type 
                          for arg_type in primitive.__annotations__.values()][1:],
            'return_type': primitive.__annotations__['return']
        }

    def sense_problem(self, problem):
        self.layers['sensation']['vision'].signal_queue.put(Signal(pid='sensor', data=problem))

    def __del__(self):
        if hasattr(self, 'global_mem'):
            self.global_mem.close()
            self.global_mem.unlink()
        for layer in self.shm.values():
            layer.close()
            layer.unlink()
