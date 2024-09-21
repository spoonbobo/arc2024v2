import inspect
import pickle

from multiprocessing.shared_memory import SharedMemory

from brain.cognitive_process import Signal
from brain.layers_def import CPs, Ls
from arc_dsl import dsl

class LRMB:

    def __init__(self):
        self.shm = {}
        self.global_mem = SharedMemory(create=True, size=64, name='global_mem')
        self.primitive_mem = None
        self.text_embedder = None # TODO
        self.initialize_primitives()
        self.layers = self.initialize_layers()
        self.initialize_connections()

    def initialize_layers(self):
        layers = {}
        for layer, layer_name in Ls.items():
            self.shm[layer_name] = SharedMemory(create=True,
                                           size=1024,
                                           name=f'{layer}_shm')
            
            layers[layer_name] = {name: process(pid=name, 
                                           layer=layer)
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
                    
    def initialize_primitives(self):
        primitives = {}
        for name, primitive in inspect.getmembers(dsl):
            if inspect.isfunction(primitive):
                primitives[name] = self.parse_primitive(primitive, name)

        pickled_data = pickle.dumps(primitives)
        pickled_size = len(pickled_data)
        self.primitive_mem = SharedMemory(create=True, size=pickled_size, name='primitive_mem')
        self.primitive_mem.buf[:] = pickled_data
    
    @staticmethod
    def parse_primitive(primitive, name):
        return {
            'name': name,
            'function': primitive,
            'arg_types': [arg_type 
                          for arg_type in primitive.__annotations__.values()][1:],
            'return_type': primitive.__annotations__['return']
        }
                    
    def activate(self):
        for layer in self.layers.values():
            for process in layer.values():
                process.start()

    def sense_problem(self, problem):
        self.layers['sensation']['vision'].signal_queue.put(Signal(pid='sensor', data=problem))

    def __del__(self):
        if hasattr(self, 'global_mem'):
            self.global_mem.close()
            self.global_mem.unlink()
        if hasattr(self, 'primitive_mem'):
            self.primitive_mem.close()
            self.primitive_mem.unlink()
        for layer in self.shm.values():
            layer.close()
            layer.unlink()
