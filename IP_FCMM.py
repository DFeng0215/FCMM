import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def initialize_matrices(n, q, c):
    """初始化子群集隸屬度矩陣 Y 和最終群集隸屬度矩陣 Z"""
    Y = np.random.rand(n, q)
    Y /= np.sum(Y, axis=1, keepdims=True)
    
    Z = np.random.rand(q, c)
    Z /= np.sum(Z, axis=0, keepdims=True)
    
    return Y, Z

def update_subcluster_centers(data, Y, q, r):
    """更新子群集中心 M"""
    um = Y ** r
    M = (data.T @ um / np.sum(um, axis=0)).T
    return M

def update_final_cluster_centers(M, Z, r):
    """更新最終群集中心 V"""
    vm = Z ** r
    V = (M.T @ vm / np.sum(vm, axis=0)).T
    return V

def update_membership_matrix(data, centers, r):
    """更新隸屬度矩陣 Y 或 Z"""
    n, c = data.shape[0], centers.shape[0]
    p = 2. / (r - 1)
    U = np.zeros((n, c))
    for i in range(n):
        x = data[i]
        dists = np.linalg.norm(x - centers, axis=1)
        U[i] = 1. / np.sum((dists[:, np.newaxis] / dists) ** p, axis=1)
    return U

def fcmm(data, q, c, r, alpha, error=0.005, maxiter=100):
    """實現 FCMM 算法"""
    n = data.shape[0]
    Y, Z = initialize_matrices(n, q, c)
    iteration = 0
    while iteration < maxiter:
        M = update_subcluster_centers(data, Y, q, r)
        V = update_final_cluster_centers(M, Z, r)
        Y_old = Y.copy()
        Z_old = Z.copy()
        Y = update_membership_matrix(data, M, r)
        Z = update_membership_matrix(M, V, r)
        if np.linalg.norm(Y - Y_old) < error and np.linalg.norm(Z - Z_old) < error:
            break
        iteration += 1
    return Y, Z, M, V

# Load Indian Pines Hyperspectral Dataset

# 高光譜影像數據，三維資料集，分別是長、寬、光譜波段
img = np.load('indianpinearray.npy')
# 地面真實值(Ground Truth)，二維的對應每個數據點的歸屬
gt = np.load('IPgt.npy')
classes = np.unique(gt)

#將三維數據降至二維
data = img.reshape(-1, img.shape[-1])

q, c = 32, 16  # Number of subclusters and clusters
r = 1.33  # Fuzzy factor
alpha = 8.3  # Balance parameter

Y, Z, M, V = fcmm(data, q, c, r, alpha)
print("Subcluster centers (M):", M)
print("Final cluster centers (V):", V)

