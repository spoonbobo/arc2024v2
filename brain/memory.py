from brain.cognitive_process import CognitiveProcess

class SBM(CognitiveProcess):
    """ sensory buffer memory """
    following = {
        '*'
    }
    followers = {
        '*'
    }
    memory = {}

    def process_signal(self, signal, result):
        if not signal.pid.startswith('vision'):
            return

        self.memory[signal.pid] = signal.data
        result.data = self.memory[signal.pid]

class ABM(CognitiveProcess):
    """ action buffer memory """
    following = {
        '*'
    }
    followers = {
        '*'
    }

    memory = {}

    def process_signal(self, signal, result):
        pass

class STM(CognitiveProcess):
    """ short-term memory """
    following = {
        '*'
    }
    followers = {
        '*'
    }
    memory = {}

    def process_signal(self, signal, result):
        pass

class LTM(CognitiveProcess):
    """ long-term memory """
    following = {
        '*'
    }
    followers = {
        '*'
    }
    memory = {}

    def process_signal(self, signal, result):
        pass

class CPM(CognitiveProcess):
    """ Not sure what this memory is for at the moment """
    following = {
        '*'
    }
    followers = {
        '*'
    }
    memory = {}

    def process_signal(self, signal, result):
        pass
