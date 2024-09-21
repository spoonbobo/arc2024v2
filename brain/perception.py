from pprint import pprint
from typing import List

import numpy as np
from z3 import *  

from brain.cognitive_process import CognitiveProcess

class Action(CognitiveProcess):
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass

class SelfConsciousness(CognitiveProcess):
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass

class Motivation(CognitiveProcess):
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass

class GoalSetting(CognitiveProcess):
    """ 
    establishes a desired and valued outcome for a motivation or an action
    """
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass

class Emotions(CognitiveProcess):
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass

class SenseOfSpatiality(CognitiveProcess):
    """ 
    generates an abstract sense of perception on the spatiality
    and allows brain to aware and assess the location of entities in space.
    
    It involves object abstraction, positioning, etc.
    """
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        result.data = signal.data

class SenseOfMotion(CognitiveProcess):
    """
    generates an abstract and internal sense of perception on motion,
    detects and interprets status changes related to space and time of external objects
    """
    following = {
        'memory:*': None,
        'meta_cognitive:*': None
    }
    followers = {
        'memory:*': None,
        'meta_cognitive:*': None
    }

    def process_signal(self, signal, result):
        pass
