import pickle
import numpy as np


with open('outputs/want/nn5_weekly.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)