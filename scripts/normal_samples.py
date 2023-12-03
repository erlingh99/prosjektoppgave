import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

num_std = 3.5 #sqrt(gamma)
Sigma = np.array([[4, 1], [1, 1]])

lambda_, eig_vals = np.linalg.eig(Sigma)
lambda_root = np.sqrt(lambda_)
width = lambda_root[0]*num_std
height = lambda_root[1]*num_std

t = np.linspace(0, 2*np.pi, 100)
x, y = eig_vals@np.diag(lambda_root)@np.array([np.cos(t), np.sin(t)])

num_points = 15 #in each direction, so actual number is this squared (minus points far away which are pruned later)

ws = np.linspace(-num_std, num_std, num_points)
hs = ws.copy()

ws, hs = np.meshgrid(ws, hs)
ws, hs = ws.ravel(), hs.ravel()

deg2rad = np.pi/180
points = np.array([[np.cos(45*deg2rad), -np.sin(45*deg2rad)],[np.sin(45*deg2rad), np.cos(45*deg2rad)]])@np.vstack((ws, hs))
# points = np.vstack((ws, hs))
idx = np.linalg.norm(points, axis=0) <= num_std #remove points outside num_std ellipsis
points = points[:, idx]
tpoints = eig_vals@np.diag(lambda_root)@points

plt.scatter(tpoints[0, :], tpoints[1, :])
plt.plot([0, eig_vals[0, 0]], [0, eig_vals[1, 0]], "r")
plt.plot([0, eig_vals[0, 1]], [0, eig_vals[1, 1]], "g")
for g in np.arange(1, num_std + 0.5, 0.5):
    plt.plot(g*x, g*y, label=f"{g}σ")
plt.axis("equal")
plt.legend()
plt.show()
