""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 3000)

with torch.no_grad():
    generator.rollout(None)
