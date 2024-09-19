from brain.cognitive_process import CognitiveProcess

class ProblemSolving(CognitiveProcess):
    following = {
        'higher_cognitive:decision_making',
        'meta_inference:*',
        'meta_cognitive:*',
        'memory:*'
    }
    followers = {
        'higher_cognitive:decision_making',
        'meta_inference:*',
        'meta_cognitive:*',
        'memory:*'
    }

    

    def process_signal(self, signal, result):
        pass

class DecisionMaking(CognitiveProcess):
    following = {
        'higher_cognitive:problem_solving',
        'meta_inference:*',
        'meta_cognitive:*',
        'memory:*'
    }
    followers = {
        'higher_cognitive:problem_solving',
        'meta_inference:*',
        'meta_cognitive:*',
        'memory:*'
    }

    

    def process_signal(self, signal, result):
        pass