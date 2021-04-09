import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('s_mat.csv')
sim = cosine_similarity(data, data)

def retSim():
    print("\n\n",(sim*30)[0],"\n\n")