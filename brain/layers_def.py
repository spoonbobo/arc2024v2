from brain.sensation import Vision
from brain.memory import SBM, STM, LTM, ABM
from brain.perception import (SelfConsciousness, 
                              Action, 
                              Motivation, 
                              GoalSetting, 
                              Emotions, 
                              SenseOfSpatiality, 
                              SenseOfMotion)
from brain.meta_cognitive import (Attention, 
                                  ConceptEstablishment, 
                                  Abstraction, 
                                  Search, 
                                  Categorization, 
                                  Memorization, 
                                  KnowledgeRepresentation)
from brain.meta_inference import (MetaCSP, Synthesis)
from brain.higher_cognitive import (ProblemSolving, DecisionMaking)

CPs = {
    'sensation': {
        'vision': Vision
    },
    'memory': {
        'sbm': SBM, 
        'ltm': LTM, 
        'stm': STM, 
        'abm': ABM
    },
    'action': {
        'action': Action
    },
    'perception': {
        'self_consciousness': SelfConsciousness, 
        'motivation': Motivation, 
        'goal_setting': GoalSetting, 
        'emotions': Emotions, 
        'sense_of_spatiality': SenseOfSpatiality, 
        'sense_of_motion': SenseOfMotion
    },
    'meta_cognitive': {
        'attention': Attention, 
        'concept_establishment': ConceptEstablishment, 
        'abstraction': Abstraction, 
        'search': Search, 
        'categorization': Categorization, 
        'memorization': Memorization, 
        'knowledge_representation': KnowledgeRepresentation
    },
    'meta_inference': {
        'csp': MetaCSP,
        'synthesis': Synthesis
    },
    'higher_cognitive': {
        # 'learning': Learning, 
        'problem_solving': ProblemSolving, 
        'decision_making': DecisionMaking
    }
}

Ls = {
    'L1': 'sensation',
    'L2': 'memory',
    'L3': 'action',
    'L4': 'perception',
    'L5': 'meta_cognitive',
    'L6': 'meta_inference',
    'L7': 'higher_cognitive'
}