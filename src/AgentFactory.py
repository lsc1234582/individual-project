"""
Create agent with specific models (e.g. neural networks with a specific architecture) and methods(e.g. actor
critic) and tunable hyperparameters (e.g. hidden layer size of the value estimator nn)

New agent types need to be registered here.
"""

import DPGAC2PEH2VEH2
import DPGAC2PEH3VEH3

from SPGAC2PEH1VEH1 import MakeSPGAC2PEH1VEH1
from DPGAC2PEH2VEH2 import MakeDPGAC2PEH2VEH2
from DPGAC2PEH3VEH3 import MakeDPGAC2PEH3VEH3
from DPGAC2WithDemoPEH2VEH2 import MakeDPGAC2WithDemoPEH2VEH2
#from DPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2 import MakeDPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2
from DPGAC2WithPrioritizedRBPEH2VEH2 import MakeDPGAC2WithPrioritizedRBPEH2VEH2
from ModelBasedMEH2 import MakeModelBasedMEH2

AGENTS = {
        "SPGAC2PEH1VEH1": MakeSPGAC2PEH1VEH1,
        "DPGAC2PEH2VEH2": MakeDPGAC2PEH2VEH2,
        "DPGAC2PEH3VEH3": MakeDPGAC2PEH3VEH3,
        "DPGAC2WithDemoPEH2VEH2": MakeDPGAC2WithDemoPEH2VEH2,
        #"DPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2": MakeDPGAC2WithMultiPModelAndDemoPEH2VEH2MEH2,
        "DPGAC2WithPrioritizedRBPEH2VEH2": MakeDPGAC2WithPrioritizedRBPEH2VEH2,
        "ModelBasedMEH2": MakeModelBasedMEH2,
        }

AGENT_PARSER = {
        "DPGAC2PEH2VEH2": DPGAC2PEH2VEH2.getArgParser,
        "DPGAC2PEH3VEH3": DPGAC2PEH3VEH3.getArgParser,
        }

def MakeAgent(session, env, args):
    return (AGENTS[args.agent_name])(session, env, args)

def GetAgentArgParser(args):
    return (AGENT_PARSER[args.agent_name])()
