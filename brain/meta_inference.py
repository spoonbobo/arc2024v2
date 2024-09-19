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
    
    z3solver: Z3ForARC = None
    
    def __init__(self, pid, layer, **kwargs):
        super().__init__(pid, layer)
        self.z3solver = Z3ForARC(**kwargs)

    def process_signal(self, signal, result):
        self.z3solver.set_variables()
        self.z3solver.set_constraints()
        self.z3solver.set_meta_rules()
        self.z3solver.solve_csp()
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
