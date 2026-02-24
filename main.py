from diffusion import diffusion
from feature_learning import feature_learning
from me_network import get_me_network
from module_detection import generate_initial_modules, module_optimization
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cancer', type=str, 
                    help='(Required) Cancer type', 
                    required=True)
parser.add_argument('-s', '--subtypes', nargs='*',
                    help="(Required) All subtypes of the cancer, including specific subtypes and subtypes used for comparison.",
                    required=True)
parser.add_argument('-p', '--specific_subtypes', type=str, 
                    help="(Optional) Subtypes used to identifying driver modules. " \
                    "If specified, only the driver modules of these specified subtypes will be identified.",
                    default=None)

args = parser.parse_args()

if (args.specific_subtypes != None):
    if (not set(args.specific_subtypes).issubset(set(args.subtypes))):
        raise ValueError("One or more specific subtypes (specified by --specific_subtypes) " \
        "are not in all subtypes used (specified by --subtypes).") 

diffusion(args.cancer, args.subtypes)

get_me_network(args.cancer, args.subtypes)


if (args.specific_subtypes == None):
    for subtype in args.subtypes:
        feature_learning(args.cancer, subtype)
        generate_initial_modules(args.cancer, subtype)
        module_optimization(args.cancer, subtype)
else:
    for subtype in args.specific_subtypes:
        feature_learning(args.cancer, subtype)
        generate_initial_modules(args.cancer, subtype)
        module_optimization(args.cancer, subtype)

print("Finished.")