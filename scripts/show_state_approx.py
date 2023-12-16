import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

##use checkboard pattern for unknown intensity, or from file
checkboard = False
## the original mean
mean = np.array([-62, -98])/3
# mean = np.array([-90, -120])/3

gz = 5 #precompute grid-size

### find R at this position
def R(mean):
    r = np.linalg.norm(mean - np.array([0, -8]))
    theta = np.arctan2(mean[1], mean[0])

    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    J = np.array([[-r*s_theta, c_theta],[r*c_theta, s_theta]])
    polar = np.array([[2*(np.pi/180)**2, 0],[0, 9]])
    return J@polar@J.T + np.eye(2)*4

sigma = R(mean) 

### create sampling grid
num_std = 4
num_points = 10*num_std #num points in each direction within the num_std ellipsis
#should really also be dependent on the size of the ellipsis (eig_vals), however assuming constant
#we can use the same discretization for all z's
spacing = 2*num_std/num_points
x = np.arange(-num_std, num_std, spacing)
x, y = np.meshgrid(x, x)
x, y = x.ravel(), y.ravel()
deg2rad = np.pi/180
points = np.array([[np.cos(45*deg2rad), -np.sin(45*deg2rad)],[np.sin(45*deg2rad), np.cos(45*deg2rad)]])@np.vstack((x, y))
idx = np.linalg.norm(points, axis=0) <= num_std #remove points outside num_std ellipsis
points = points[:, idx]

### compute the standard normal density at the values
normal_values = norm.pdf(points[0, :])*norm.pdf(points[1, :]) #normalized sample points values

### eigendecomp and points tranform
eig_vals, eig_vecs = np.linalg.eig(sigma)
eig_root = np.diag(np.sqrt(eig_vals))

tpoints = eig_vecs@eig_root@points + mean[..., np.newaxis]

### get/create birth intensity
if checkboard:
    gs = 10
    lim = 4*np.sqrt(max(eig_vals)) + gs//2
    g = np.arange(-lim, lim, gs) + gs//2
    g1, g2 = np.meshgrid(g, g)
    grid = np.vstack((g1.ravel(), g2.ravel())) + np.round(mean[..., None], -1)
    c = np.sum(grid.T//gs, axis=-1)%2
    birth_intensity = np.hstack((grid.T, c[..., np.newaxis]))
else:
    birth_intensity = np.load("./birth_analysis/custom_intensity_grid_2.npy")
    gs = 10
    lim = 4*np.sqrt(max(eig_vals)) + gs//2

us = np.zeros((len(tpoints.T)))
for i, tp in enumerate(tpoints.T):
    pos_diff = np.linalg.norm(birth_intensity[:, :2] - tp, axis=-1)
    min_diff_idx = np.argmin(pos_diff)
    us[i] = np.where(birth_intensity[min_diff_idx, 2], 1e-5, 1e-6)

### compute transformed values
integral = spacing**2*us@normal_values
nmean = spacing**2*(us*tpoints)@normal_values/integral
err = tpoints - nmean[..., np.newaxis]
err_outer = np.einsum("ji,ki -> ijk", err, err)
ue = np.einsum("i,ijk -> ijk", us, err_outer)
nsigma = spacing**2*np.einsum("ijk, i -> jk", ue, normal_values)/integral

neig_vals, neig_vecs = np.linalg.eig(nsigma)
neig_root = np.diag(np.sqrt(neig_vals))

### compute average
temp = np.vstack((mean//gz*gz - gz/2 , (mean//gz + 1)*gz - gz/2))
mx, my = np.meshgrid(temp[:, 0], temp[:, 1])
mx, my = mx.ravel(), my.ravel()
means = np.vstack((mx, my)).T
print(means)

d = np.linalg.norm(mean - means, axis=-1)
if min(d) < 1e-2:
    means = np.array([mean])
    ws = np.array([1])
else:
    ws = 1/d

ameans = np.zeros_like(means)
asigmas = np.zeros((len(means), 2, 2))
for j, m in enumerate(means):
    asigma = R(m)
    aev, aevc = np.linalg.eig(asigma)
    aer = np.diag(np.sqrt(aev))
    tpoints = aevc@aer@points + m[..., np.newaxis]

    us = np.zeros((len(tpoints.T)))
    for i, tp in enumerate(tpoints.T):  
        pos_diff = np.linalg.norm(birth_intensity[:, :2] - tp, axis=-1)
        min_diff_idx = np.argmin(pos_diff)
        us[i] = np.where(birth_intensity[min_diff_idx, 2], 1e-5, 1e-6)
    integral = spacing**2*us@normal_values
    am = spacing**2*(us*tpoints)@normal_values/integral
    err = tpoints - am[..., np.newaxis]
    err_outer = np.einsum("ji,ki -> ijk", err, err)
    ue = np.einsum("i,ijk -> ijk", us, err_outer)
    ansigma = spacing**2*np.einsum("ijk, i -> jk", ue, normal_values)/integral
    ameans[j] = am
    asigmas[j] = ansigma

amean = np.average(ameans, axis=0, weights=ws)
asigma = np.average(asigmas, axis=0, weights=ws)

expects_outer = np.einsum("ij,ik -> ijk", ameans, ameans)
wP = np.average(expects_outer, axis=0, weights=ws)
asigma_spread = wP - np.outer(amean, amean)

asigma = asigma #+ asigma_spread
aeigs, aeig_vecs = np.linalg.eig(asigma)
aeig_root = np.diag(np.sqrt(aeigs))

### setup plot
fig = plt.figure()
ax1, ax2, ax3 = fig.subplots(1, 3)

### axes
std_axes = np.array([[0, 0, 1],
                    [0, 1, 0]])
### ellipsis plot angles
t = np.linspace(0, 2*np.pi, 100)

### load map
with open("./ravnkloa_radar_base_map.yaml", "r") as f:
    doc = yaml.safe_load(f)
map = np.fromfile("./ravnkloa_radar_base_map.bin", dtype=bool).reshape((doc["map_height"], doc["map_width"])).T

titles = ["Initial normal distribution", "Exact", "Averaged"]
for ti, (ax, m, ev, er) in enumerate(((ax1, mean, eig_vecs, eig_root), (ax2, nmean, neig_vecs, neig_root), (ax3, amean, aeig_vecs, aeig_root))):
    ### transform and plot axes
    a = ev@er@std_axes + m[:, np.newaxis]
    ax.plot([a[0, 0], a[0, 1]], [a[1, 0], a[1, 1]], "r")
    ax.plot([a[0, 0], a[0, 2]], [a[1, 0], a[1, 2]], "g")

    ### plot std ellipses
    tx, ty = ev@er@np.array([np.cos(t), np.sin(t)])
    for g in np.arange(1, num_std + 0.5, 1):
        ax.plot(g*tx + m[0], g*ty + m[1], label=f"{g:1.0f}σ")

    ### plot birth intensity grid
    for i, (xx, yy, cc) in enumerate(birth_intensity):
        d = np.linalg.norm([xx, yy] - mean)
        if cc and d < lim:
            r = Rectangle((xx - gs/2, yy - gs/2), gs, gs, color=(0.1, 0.1, 0.1, 0.3))
            ax.add_patch(r)
        elif d < lim:
            r = Rectangle((xx - gs/2, yy - gs/2), gs, gs, color=(1, 1, 1, 0.3))
            ax.add_patch(r)

    ### plot map
    ax.imshow(map, origin="lower",   
            extent=[doc["origin_y_coordinate"],
                    doc["origin_y_coordinate"] + doc["map_height"],
                    doc["origin_x_coordinate"],
                    doc["origin_x_coordinate"] + doc["map_width"]],
            cmap="BuGn",
            vmin=-1,#to get a bit less blue water
            vmax=2)#to get a bit less green land
    ### plot radar
    ax.scatter(0, -8, c="r", marker="x", label="Radar")
    ax.legend()
    ax.set_title(titles[ti])
    ax.set_xlim(-100, 100)
    ax.set_ylim(-150, 50)

print(sigma)
print(nsigma)
print(asigma)
print("===")
print(mean)
print(nmean)
print(amean)




plt.show()