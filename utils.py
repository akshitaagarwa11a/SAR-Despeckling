import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict 
import datetime
from scipy import special
import cv2


def lintolog(image, max_val=1.0, min_val=-1.0):
    image = image.numpy()
    LIN_MAX = 1.0
    LOG_MAX = np.log10(LIN_MAX + 1)
    image, dtype = _get_float32(image)
    img_min = np.min(image)
    img_max = np.max(image)
    stretch_max = LIN_MAX * (img_max - min_val) / (max_val - min_val)
    stretch_min = LIN_MAX * (img_min - min_val) / (max_val - min_val)
    image = stretch(image, max_val=stretch_max, min_val=stretch_min)
    image = np.log10(image + 1)
    image = image / LOG_MAX
    image = _revert_dtype(image, dtype)
    return torch.DoubleTensor(image)

def logtolin(image, max_val=1.0, min_val=0):
    image = image.detach().numpy()
    LIN_MAX = 1.0
    LOG_MAX = np.log10(LIN_MAX + 1)
    image, dtype = _get_float32(image)
    img_min = np.min(image)
    img_max = np.max(image)
    stretch_max = LOG_MAX * (img_max - min_val) / (max_val - min_val)
    stretch_min = LOG_MAX * (img_min - min_val) / (max_val - min_val)
    image = stretch(image, max_val=stretch_max, min_val=stretch_min)
    image = np.power(10, image) - 1
    image = image / LIN_MAX 
    image = _revert_dtype(image, dtype)
    return torch.DoubleTensor(image)

def simulate_speckle(clean_im, L):
    M = np.log(256)
    m = 0
    s = torch.zeros_like(clean_im)
    for k in range(0, L):
        gamma1 = torch.normal(mean=0, std=1, size=clean_im.size())**2
        gamma2 = torch.normal(mean=0, std=1, size=clean_im.size())**2
        s = s + torch.abs(gamma1 + gamma2)
    s_amplitude = torch.sqrt(s / L)
    log_speckle = torch.log(s_amplitude)
    log_norm_speckle = log_speckle / (M - m)
    noisy_im = clean_im + log_norm_speckle
    noisy_im = torch.clamp(noisy_im,min=0.0, max = 1.0)
    return noisy_im