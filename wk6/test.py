import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--model', default='/model/', type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

protocol = '/data/tiny-imagenet-200/words.txt'

with open(protocol) as f:
    id_label_lines = f.readline()

for line in id_label_lines:
    l = line.replace('\n','').split('\t')
    todo

    