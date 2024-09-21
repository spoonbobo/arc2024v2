from z3 import *

from data_loader import ARCProblem

class Z3ForARC:

    def __init__(self, 
                 primitives: dict):
        self.primitives = primitives
        self.num_pairs = None
        self.s = None
        self.T = None
        self.P = None
        self.max_steps = None
        self.grid_shapes = None
        
    def initialize(self,
                   problem: ARCProblem,
                   max_steps: int = 20):
        self.s = Solver()
        self.num_pairs = len(problem)
        self.max_steps = max_steps
        self.grid_shapes = [{} for _ in range(self.num_pairs)]

        self.T = [[Int(f'T_{k}_{i}') 
                   for i in range(max_steps)] 
                  for k in range(self.num_pairs)]

        self.P = [[{} for _ in range(max_steps)] 
                  for k in range(self.num_pairs)]


if __name__ == '__main__':
    z3 = Z3ForARC()
