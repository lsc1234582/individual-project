"""
Create agent with specific models (e.g. neural networks with a specific architecture) and methods(e.g. actor
critic) and tunable hyperparameters (e.g. hidden layer size of the value estimator nn)

New agent types need to be registered here.
"""

from SPGAC2PEH1VEH1 import MakeSPGAC2PEH1VEH1

AGENTS = {
        "SPGAC2PEH1VEH1": MakeSPGAC2PEH1VEH1,
        }

def MakeAgent(session, env, args):
    return (AGENTS[args.agent])(session, env, args)
