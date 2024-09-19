import json
import os
import random
import re
import inspect
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from arc_dsl.arc_types import *
from arc_dsl import constants
from arc_dsl import solvers
from plot import create_task_canvas, render_task
from re_arc import generators

def grid_to_tuple(grid: List[List[int]]) -> Grid:
    """Convert a grid (list of lists) to a tuple of tuples."""
    return tuple(tuple(row) for row in grid)

@dataclass
class ARCAnswer:
    function_calls: List[str]
    source: List[str]
    traces: Dict[int, List[Dict[str, Any]]]
    reasoning_levels: Dict[int, List[str]]

    def __len__(self):
        return len(self.reasoning_levels)

    def __getitem__(self, key):
        return self.function_calls[key]

class ARCProblem:
    def __init__(self, problem_id: str, train_inputs: List[List[List[int]]], train_outputs: List[List[List[int]]], 
                 test_inputs: List[List[List[int]]], test_solutions: List[List[List[int]]], metadata: Dict[str, Any] = None):
        self.problem_id = problem_id
        self.train_inputs = [grid_to_tuple(grid) for grid in train_inputs]
        self.train_outputs = [grid_to_tuple(grid) for grid in train_outputs]
        if test_inputs is not None:
            self.test_inputs = [grid_to_tuple(grid) for grid in test_inputs]
        else:
            self.test_inputs = None
        if test_solutions is not None:
            self.test_solutions = [grid_to_tuple(grid) for grid in test_solutions]
        else:
            self.test_solutions = None
    
    def __str__(self):
        return f'ARCProblem {self.problem_id}'

    def __len__(self):
        return len(self.train_inputs)

    def visualize(self):
        canvas = create_task_canvas(self.train_inputs,
                                    self.train_outputs,
                                    grid_width=190,
                                    vertical_spacing=20,
                                    draw_colors=True)
        render_task(canvas)

class ARCDataset:
    
    def __init__(self, base_path: str, subset='train'):
        self.base_path = base_path
        self.subset = subset
        train_challenges = self.load_json(os.path.join(base_path, 'arc-agi_training_challenges.json'))
        train_solutions = self.load_json(os.path.join(base_path, 'arc-agi_training_solutions.json'))
        evaluation_challenges = self.load_json(os.path.join(base_path, 'arc-agi_evaluation_challenges.json'))
        evaluation_solutions = self.load_json(os.path.join(base_path, 'arc-agi_evaluation_solutions.json'))
        
        self.data = {}
        
        if subset == 'train':
            self.data = self.load_training_data(train_challenges, train_solutions)
        else:
            self.data = self.load_evaluation_data(evaluation_challenges, evaluation_solutions)
        
    def load_training_data(self, training_challenges, training_solutions):
        data = {}
        for key, task in training_challenges.items():
            train_inputs = [example['input'] for example in task['train']]
            train_outputs = [example['output'] for example in task['train']]
            test_inputs = [example['input'] for example in task['test']]
            test_solutions = [training_solutions[key][i] for i in range(len(test_inputs))]
            problem = ARCProblem(key, train_inputs, train_outputs, test_inputs, test_solutions)
            data[key] = problem
        
        return data

    def load_evaluation_data(self, evaluation_challenges, evaluation_solutions):
        data = {}
        for key, task in evaluation_challenges.items():
            train_inputs = [example['input'] for example in task['train']]
            train_outputs = [example['output'] for example in task['train']]
            test_inputs = [example['input'] for example in task['test']]
            test_solutions = [evaluation_solutions[key][i] for i in range(len(test_inputs))]
            problem = ARCProblem(key,  train_inputs, train_outputs, test_inputs, test_solutions)
            data[key] = problem
        
        return data
    
    def __getitem__(self, task: str):
        return self.data[task]
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data.values())
    
    def __str__(self):
        return self.name
    
    @staticmethod
    def load_json(fp):
        with open(fp, 'r') as f:
            return json.load(f)

    def sample(self) -> ARCProblem:

        return random.choice(list(self.data.values()))

class ReARCDataset(ARCDataset):
    curriculum = {
        'easy-rotation': [
            '3c9b0459',
            '6150a2bd',
            'ed36ccf7'
        ],
        
        'level0': [
            "67a3c6ac",
            "68b16354",
            "74dd1130",
            "3c9b0459",
            "6150a2bd",
            "9172f3a0",
            "9dfd6313",
            "a416b8f3",
            "b1948b0a",
            "c59eb873",
            "c8f0f002",
            "d10ecb37",
            "d511f180",
            "ed36ccf7",
            "4c4377d9",
            "6d0aefbc",
            "6fa7a44f",
            "5614dbcf",
            "5bd6f4ac",
            "5582e5ca",
            "8be77c9e",
            "c9e6f938",
            "2dee498d"
        ],

        'level1': [
            '4c4377d9',
            '6d0aefbc',
            '6fa7a44f',
            '5614dbcf',
            '5bd6f4ac',
            '5582e5ca',
            '8be77c9e',
            'c9e6f938',
            '2dee498d',
            '1cf80156',
            '32597951',
            '25ff71a9',
            '0b148d64',
            '1f85a75f',
            '23b5c85d',
            '9ecd008a',
            'ac0a08a4',
            'be94b721',
            'c909285e',
            'f25ffba3',
            'c1d99e64',
            'b91ae062',
            '3aa6fb7a',
            '7b7f7511',
            '4258a5f9'
        ],

        'level2': [
            '2dc579da',
            '28bf18c6',
            '3af2c5a8',
            '44f52bb0',
            '62c24649',
            '67e8384a',
            '7468f01a',
            '662c240a',
            '42a50994',
            '56ff96f3',
            '50cb2852',
            '4347f46a',
            '46f33fce',
            'a740d043',
            'a79310a0',
            'aabf363d',
            'ae4f1146',
            'b27ca6d3',
            'ce22a75a',
            'dc1df850',
            'f25fbde4',
            '44d8ac46',
            '1e0a9b12',
            '0d3d703e',
            '3618c87e',
            '1c786137',
            '8efcae92',
            '445eab21',
            '6f8cd79b',
            '2013d3e2',
            '41e4d17e',
            '9565186b',
            'aedd82e4',
            'bb43febb',
            'e98196ab',
            'f76d97a5',
            'ce9e57f2',
            '22eb0ac0',
            '9f236235',
            'a699fb00'
        ],

        'level3': [
            '46442a0e', '7fe24cdd', '0ca9ddb6', '543a7ed5', '0520fde7',
            'dae9d2b5', '8d5021e8', '928ad970', 'b60334d2', 'b94a9452',
            'd037b0a7', 'd0f5fe59', 'e3497940', 'e9afcf9a', '48d8fb45',
            'd406998b', '5117e062', '3906de3d', '00d62c1b', '7b6016b9',
            '67385a82', 'a5313dff', 'ea32f347', 'd631b094', '10fcaaa3',
            '007bbfb7', '496994bd', '1f876c06', '05f2a901', '39a8645d',
            '1b2d62fb', '90c28cc7', 'b6afb2da', 'b9b7f026', 'ba97ae07',
            'c9f8e694', 'd23f8c26', 'd5d6de2d', 'dbc1a6ce', 'ded97339',
            'ea786f4a', '08ed6ac7', '40853293', '5521c0d9', 'f8ff0b80',
            '85c4e7cd'
        ]
    }
    def __init__(self, 
                 problems: List[str] = None, 
                 curriculum: str | list = None,
                 n_pairs: int = 5,
                 diff_lb: float = 0,
                 diff_ub: float = 0.1,
                 total_pairs: int = 1000):
        
        def parse_generator_functions(module):
            problem_ids = []
            
            # Iterate through all members of the module
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and name.startswith("generate_"):
                    problem_id = name[len("generate_"):]
                    problem_ids.append(problem_id)
            return problem_ids

        self.generator = {name: func for name, func in inspect.getmembers(generators, inspect.isfunction) if name.startswith('generate_')}
        self.n_pairs = n_pairs
        self.diff_lb = diff_lb
        self.diff_ub = diff_ub
        self.data = {}
        
        if curriculum is not None:
            if isinstance(curriculum, str):
                problems = self.curriculum.get(curriculum, [])
            elif isinstance(curriculum, list):
                problems = []
                for c in curriculum:
                    problems += self.curriculum.get(c, [])

        if problems is None and curriculum is None:
            problems = parse_generator_functions(generators)

        if not problems:
            raise ValueError(f"No problems are loaded.")
            
        self.name = f'ReARCDataset: problems {problems} (curriculum: {curriculum})'
        self.puzzles = problems

        self.current_puzzle = None
        self.puzzle_id = 0
        
        self.solutions = self.extract_solvers()
    
    def extract_solvers(self):
        solver_info = {}
        
        for name, obj in inspect.getmembers(solvers):
            if name.startswith('solve_') and inspect.isfunction(obj):
                source_lines, _ = inspect.getsourcelines(obj)
                problem_id = name.split('_')[1]
                function_calls = {}
                for line in source_lines[1:-1]:
                    match = re.match(r'\s*(\w+)\s*=\s*(\w+)\((.*)\)', line)
                    if match:
                        var_name, func_name, args = match.groups()
                        args_list = [arg.strip() for arg in args.split(',')]
                        function_calls[var_name] = {
                            'function': func_name,
                            'args': args_list
                        }
                
                solver_info[problem_id] = {
                    'function_calls': function_calls,
                    'source': source_lines
                }
        
        return solver_info

    def assign_puzzle(self):
        self.current_puzzle = random.choice(self.puzzles)
        self.puzzle_id = 0
        
    def get_solution(self, problem: ARCProblem) -> ARCAnswer:
        puzzle_name = problem.problem_id.split('-')[0]
        target_priors = self.solutions[puzzle_name]
        function_calls = target_priors['function_calls']
        source = target_priors['source']
        return ARCAnswer(function_calls=function_calls,
                         source=source,
                         traces=None,
                         reasoning_levels=None)

    def create_puzzle(self) -> Tuple[ARCProblem, ARCAnswer]:
        puzzle_master = self.generator['generate_'+self.current_puzzle]
        key = f'{self.current_puzzle}-{self.puzzle_id}'
        
        puzzles = [puzzle_master(self.diff_lb, self.diff_ub) for _ in range(self.n_pairs)]

        arc_problem = ARCProblem(key, 
                                train_inputs=[pair['input'] for pair in puzzles],
                                train_outputs=[pair['output'] for pair in puzzles],
                                test_inputs=None,
                                test_solutions=None)
        
        arc_solution = self.get_solution(arc_problem)
    
        self.puzzle_id += 1
        return arc_problem, arc_solution
