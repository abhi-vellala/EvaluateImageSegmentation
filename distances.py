import numpy as np
from scipy import spatial

def euclidean(A, B):
    return np.sqrt(np.sum(np.square(A - B)))

def manhattan(A, B):
    return np.sum(np.abs(A - B))

def cosine(A, B):
    dot_prod = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_prod/(norm_a * norm_b)