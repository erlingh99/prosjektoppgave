import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sepfir2d
import yaml

##import intensities
birth_int = np.load("../birth_analysis/custom_intensity_grid.npy")
blow = 1e-6
def find_birth(pos, high=1e-5, low=1e-6):
    if pos.shape == (2,):
        pos = pos[..., np.newaxis]
    pos_diff = np.linalg.norm(birth_int[:, :2] - pos.T[:, np.newaxis, ...], axis=-1)
    min_diff_idx = np.argmin(pos_diff, axis=-1)
    return np.where(birth_int[min_diff_idx, 2], high, low)

intensity = np.load("../clutter_analysis/custom_intensity_grid.npy")
def find_clutter(pos, low=1e-4, high=5e-3):
    if pos.shape == (2,):
        pos = pos[..., np.newaxis]
    pos_diff = np.linalg.norm(intensity[:, :2] - pos.T[:, np.newaxis, ...], axis=-1)
    min_diff_idx = np.argmin(pos_diff, axis=-1)

    return np.where(intensity[min_diff_idx, 2], high, low)

## import precomputed integral values
integral = np.load("../integral_vals.npy")
## extract the positions the integral is evaluated at, get birth and clutter there
pos = integral[:, :2]
integral_vals = integral[:, 2]
birth_vals = find_birth(pos.T)
clutter_vals = find_clutter(pos.T)

#reshape birth_vals to grid for convolution
resolution = int(np.linalg.norm(pos[0] - pos[1]))
h = int(max(pos[:, 1]) - min(pos[:, 1]))
w = int(max(pos[:, 0]) - min(pos[:, 0]))

birth_grid = birth_vals.reshape(h//resolution + 1, w//resolution + 1)

# kernel = np.array([1])
kernel = np.array([1, 2, 1])
# kernel = np.array([1, 4, 6, 4, 1])
kernel = kernel/sum(kernel)
smooth = sepfir2d(birth_grid, kernel, kernel).ravel()

## compute the difference it has in existence probability
P_D = 0.7
r_approx = P_D*birth_vals/(clutter_vals + P_D*birth_vals) 
r_smooth = P_D*smooth/(clutter_vals + P_D*smooth) 
r_integral = P_D*integral_vals/(clutter_vals + P_D*integral_vals) 

print(np.linalg.norm(integral_vals - smooth))
print(np.linalg.norm(integral_vals - birth_vals))
print(np.linalg.norm(r_integral - r_approx))
print(np.linalg.norm(r_integral - r_smooth))

idx = np.max([smooth, birth_vals, integral_vals], axis = 0) > blow
pos = pos[idx]
birth_vals = birth_vals[idx]
smooth = smooth[idx]
integral_vals = integral_vals[idx]
r_approx = r_approx[idx] 
r_smooth = r_smooth[idx]
r_integral = r_integral[idx]
x = pos[:, 0]
y = pos[:, 1]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
fig2 = plt.figure()
ax3 = fig2.add_subplot(1,1,1, projection="3d")

ax1.scatter(x, y, birth_vals, label="birth intensity")
ax1.scatter(x, y, smooth, label="smoothed birth intensity")
ax1.set_xlim(-200, 200)
ax1.set_ylim(-200, 200)
ax1.view_init(45, -90, 0)

ax2.scatter(x, y, birth_vals, label="birth intensity")
ax2.scatter(x, y, integral_vals, label="integral")
ax2.set_xlim(-200, 200)
ax2.set_ylim(-200, 200)
ax2.view_init(45, -90, 0)

ax3.scatter(x, y, r_approx, label="approx")
ax3.scatter(x, y, r_smooth, label="smooth")
ax3.scatter(x, y, r_integral, label="integral")
ax3.set_xlim(-200, 200)
ax3.set_ylim(-200, 200)
ax3.view_init(45, -90, 0)

with open("../ravnkloa_radar_base_map.yaml", "r") as f:
    doc = yaml.safe_load(f)

map = np.fromfile("../ravnkloa_radar_base_map.bin", dtype=bool).reshape((doc["map_height"], doc["map_width"])).T
map = map.astype(float)
mapsize = 400
#discard areas outside of mapsize/2 from center
cx = int(-doc["origin_x_coordinate"])
cy = int(-doc["origin_y_coordinate"])
map = map[cx-mapsize//2:cx+mapsize//2, cy-mapsize//2:cy+mapsize//2]

# stride arg determines image quality 
stride = 5
stepX, stepY = mapsize / (map.shape[1]/stride), mapsize / (map.shape[0]/stride)

X1 = np.arange(-mapsize/2, mapsize/2, stepX)
Y1 = np.arange(-mapsize/2, mapsize/2, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
ax1.scatter(X1, Y1, np.zeros_like(X1) + blow, c=map[::stride,::stride], cmap="BuGn", vmin=-1, vmax=2)
ax2.scatter(X1, Y1, np.zeros_like(X1) + blow, c=map[::stride,::stride], cmap="BuGn", vmin=-1, vmax=2)
ax3.scatter(X1, Y1, np.zeros_like(X1), c=map[::stride,::stride], cmap="BuGn", vmin=-1, vmax=2)

ax1.set_title("Gaussian kernel")
ax2.set_title("Integral")
ax3.set_title("Initial existence probability")
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()