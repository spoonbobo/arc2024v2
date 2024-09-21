# https://www.santanderopenacademy.com/en/blog/cognitive-processes.html
from abc import ABC, abstractmethod
import pickle
from typing import Any
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory


class Signal(ABC):
    def __init__(self, pid: str, data: Any):
        self.pid: str = pid
        self.data: Any = data

    def __repr__(self):
        return f"Signal(pid='{self.pid}', data={self.data})"

class CognitiveProcess(ABC):
    
    following: dict
    followers: dict

    def __init__(self, pid, layer):
        self.pid = pid
        self.layer = layer
        self.shm = SharedMemory(name=f'{self.layer}_shm')
        self.global_mem = SharedMemory(name='global_mem')
        self.subscribers = {}
        self.signal_queue = Queue()
        self.process = Process(target=self.loop)
    
    def start(self):
        self.process.start()

    def loop(self):
        while True:
            self.consume()

    def consume(self):
        while not self.signal_queue.empty():
            result = Signal(pid=self.pid, data=None)
            signal = self.signal_queue.get()
            if signal.data is None:
                continue
            self.process_signal(signal, result)
            self.produce(result)

    def produce(self, result):
        for _, signal_queue in self.subscribers.items():
            signal_queue.put(result)
            
        
    def load_primitives(self):
        primitive_mem = SharedMemory(name='primitive_mem')
        pickled_data = bytes(primitive_mem.buf)
        primitives = pickle.loads(pickled_data)
        return primitives

    @abstractmethod
    def process_signal(self, signal, result, *args, **kwargs):
        pass
    
    def __repr__(self):
        return f'{self.pid}-({self.layer})'

    def __str__(self):
        return self.__repr__()
