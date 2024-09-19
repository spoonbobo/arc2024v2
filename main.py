from brain.lrmb import LRMB
from data_loader import ReARCDataset

if __name__ == '__main__':
    lrmb = LRMB()
    dataset = ReARCDataset(curriculum='easy-rotation',
                       diff_lb=0.0,
                       diff_ub=0.1,
                       n_pairs=2)
    dataset.assign_puzzle()
    problem, answer = dataset.create_puzzle()
    lrmb.activate()
    lrmb.sense_problem(problem)
