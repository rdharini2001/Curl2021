import torch
import torch.nn as nn             # module for creating and training neural networks
import torch.nn.functional as F   #for improving net efficiency
from torch.utils.data import DataLoader  # represents an ierable over a dataset for loading data in batches
from torchvision import datasets, transforms 
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix # for evaluating results
import time
torch.manual_seed(101)  # for consistent results
