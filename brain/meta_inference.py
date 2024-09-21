from pprint import pprint

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
            
        pprint(data)
        pass

class ProgramNode:
    def __init__(self, primitive_name, args):
        self.primitive_name = primitive_name
        self.args = args  # Can be literals, variables, or other ProgramNodes

    def evaluate(self, context):
        primitive = primitives[self.primitive_name]['function']
        evaluated_args = [
            arg.evaluate(context) if isinstance(arg, ProgramNode) else arg
            for arg in self.args
        ]
        return primitive(*evaluated_args)

    def get_return_type(self):
        return primitives[self.primitive_name]['return_type']
        

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
    
    def generate_programs(max_depth, return_type):
        if max_depth == 0:
            return []
        programs = []
        for prim_name, prim_info in primitives.items():
            if prim_info['return_type'] != return_type:
                continue

            arg_types = prim_info['arg_types']
            arg_program_lists = [
                generate_programs(max_depth - 1, arg_type) for arg_type in arg_types
            ]
            for args in product(*arg_program_lists):
                program = ProgramNode(prim_name, args)
                programs.append(program)
        return programs
