"""
Create agent with specific models (e.g. neural networks with a specific architecture) and methods(e.g. actor
critic) and tunable hyperparameters (e.g. hidden layer size of the value estimator nn)

New agent types need to be registered here.
"""

from DPGAC2PEH2VEH2 import MakeDPGAC2PEH2VEH2
from DPGAC2WithPrioritizedRBPEH2VEH2 import MakeDPGAC2WithPrioritizedRBPEH2VEH2
from TD3PEH2VEH2 import MakeTD3PEH2VEH2
from TD3HERPEH2VEH2 import MakeTD3HERPEH2VEH2
from ModelBasedMEH2 import MakeModelBasedMEH2

AGENTS = {
        "DPGAC2PEH2VEH2": MakeDPGAC2PEH2VEH2,
        "DPGAC2WithPrioritizedRBPEH2VEH2": MakeDPGAC2WithPrioritizedRBPEH2VEH2,
        "ModelBasedMEH2": MakeModelBasedMEH2,
        "TD3PEH2VEH2": MakeTD3PEH2VEH2,
        "TD3HERPEH2VEH2": MakeTD3HERPEH2VEH2,
        }


def MakeAgent(session, env, args):
    return (AGENTS[args.agent_name])(session, env, args)

def GetAgentArgParser(args):
    return (AGENT_PARSER[args.agent_name])()
