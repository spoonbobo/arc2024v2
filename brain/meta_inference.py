from brain.cognitive_process import CognitiveProcess
from brain.prover.solver import Z3ForARC

class MetaCSP(CognitiveProcess):
    """ z3-solver based CSP """
    following = {
        'meta_cognitive:*': None,
        'higher_cognitive:*': None,
        'memory:*': None
    }
    followers = {
        'meta_cognitive:*': None,
        'higher_cognitive:*': None,
        'memory:*': None
    }
        
    def __init__(self, pid, layer, **kwargs):
        super().__init__(pid, layer)
        self.z3solver = None

    def process_signal(self, signal, result):
        data = signal.data
        if self.z3solver is None and 'problem' in data:
            self.z3solver = Z3ForARC()
            self.z3solver.initialize(data['problem'])
        pass

class Synthesis(CognitiveProcess):
    following = {
        'meta_cognitive:*': None,
        'higher_cognitive:*': None,
        'memory:*': None
    }
    followers = {
        'meta_cognitive:*': None,
        'higher_cognitive:*': None,
        'memory:*': None
    }

    def process_signal(self, signal, result):
        pass
