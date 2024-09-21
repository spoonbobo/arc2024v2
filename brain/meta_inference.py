from collections import defaultdict
from itertools import product

from brain.cognitive_process import CognitiveProcess
from brain.prover.solver import Z3ForARC
from arc_dsl.arc_types import *

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
        self.primitives = self.load_primitives()
        
    def process_signal(self, signal, result):
        data = signal.data
        if self.z3solver is None and 'problem' in data:
            self.z3solver = Z3ForARC(primitives=self.primitives)
            self.z3solver.initialize(data['problem'])
            
        pass

class ProgramNode:
    def __init__(self, primitive_name, args, func, return_type):
        self.primitive_name = primitive_name
        self.args = args  # Can be literals, variables, or other ProgramNodes
        self.func = func
        self.return_type = return_type
    
    def evaluate(self, context):
        evaluated_args = [
            arg.evaluate(context) if isinstance(arg, ProgramNode) else arg
            for arg in self.args
        ]
        return self.func(*evaluated_args)

    def get_return_type(self):
        return self.return_type

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

    def __init__(self, pid, layer, **kwargs):
        super().__init__(pid, layer)
        self.primitives = self.load_primitives()
        self.base_programs = {}

    def process_signal(self, signal, result):
        # print(self.generate_programs(2, Integer))
        self.setup_base_programs(signal)
        pass
    
    def setup_base_programs(self, signal):
        print(signal)
        pass
        
    def generate_programs(self, max_depth, return_type):
        programs_by_depth = defaultdict(list)
        
        # Initialize with base programs
        programs_by_depth[0] = self.generate_base_programs(return_type)
        
        for depth in range(1, max_depth + 1):
            for prim_name, prim_info in self.primitives.items():
                if prim_info['return_type'] != return_type:
                    continue

                arg_types = prim_info['arg_types']
                arg_program_lists = [
                    [prog for d in range(depth) for prog in programs_by_depth[d] if prog.get_return_type() == arg_type]
                    for arg_type in arg_types
                ]
                
                for args in product(*arg_program_lists):
                    program = ProgramNode(prim_name, 
                                          args, 
                                          func=prim_info['function'],
                                          return_type=prim_info['return_type'])
                    programs_by_depth[depth].append(program)
        
        return [prog for depth in programs_by_depth for prog in programs_by_depth[depth]]
