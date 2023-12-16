import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import yaml

#grid to update
# file = "./birth_analysis/custom_birth.npy"
file = "./birth_analysis/custom_intensity_grid_2.npy"
# file = "./birth_analysis/birth_intensity.npy"
# file = "./clutter_analysis/custom_intensity_grid.npy"
# file = "./clutter_analysis/clutter_intensity_all_thresholded.npy"

rx = 0 #radar pos x
ry = -8 #radar pos y

grid_size = 10 #meters
grid_radius = 150 #150 meters in each direction from radar, so 300x300 meter
grid_shape = (2*grid_radius)//grid_size
grid = np.zeros(shape=(grid_shape, grid_shape))

fig = plt.figure()
ax = fig.add_subplot(111)

with open("./ravnkloa_radar_base_map.yaml", "r") as f:
    doc = yaml.safe_load(f)

map = np.fromfile("./ravnkloa_radar_base_map.bin", dtype=bool).reshape((doc["map_height"], doc["map_width"])).T

ax.imshow(map, origin="lower",   
        extent=[doc["origin_y_coordinate"],
                doc["origin_y_coordinate"] + doc["map_height"],
                doc["origin_x_coordinate"],
                doc["origin_x_coordinate"] + doc["map_width"]],
        cmap="BuGn",
        vmin=-1,#to get a bit less blue water
        vmax=2)

for p in range(-grid_radius, grid_radius+1, grid_size):
    ax.plot([p + rx, p + rx], [-grid_radius + ry, grid_radius + ry], "k")
    ax.plot([-grid_radius + rx, grid_radius + rx], [p+ry, p + ry], "k")

table_row = np.array(range(-grid_radius, grid_radius, grid_size)) + grid_size/2
xpos, ypos = np.meshgrid(table_row, table_row, indexing="ij")
xpos = xpos.ravel() + rx
ypos = ypos.ravel() + ry
table = np.vstack((xpos, ypos, np.zeros_like(xpos))).T

if file is not None:
    init_vals = np.load(file)
    #grid might be different, so need to cast the problem to this grid
    for val in init_vals:
        pos_diff = np.linalg.norm(table[:, :2] - val[:2], axis=-1)
        idx = np.argmin(pos_diff)
        table[idx, 2] = val[2]


rectangles = [0]*len(table)
for i in range(len(table)):
    if table[i, 2] == 1:
        x, y = table[i, :2] - grid_size/2
        rect = patch.Rectangle([x, y], grid_size, grid_size, facecolor=(1, 0, 0, 0.5))
        patch_r = ax.add_patch(rect)
        rectangles[i] = patch_r
    else:
        rectangles[i] = None

def onclick(event):
    ix, iy = event.xdata, event.ydata

    if ix == None or iy == None:
        return

    if np.linalg.norm([ix - rx, iy - ry]) > grid_radius:
        return
    
    pos_diff = np.linalg.norm(table[:, :2] - [ix, iy], axis=-1)
    idx = np.argmin(pos_diff)

    if table[idx, 2] == 1:
        table[idx, 2] = 0
        rectangles[idx].remove()
        rectangles[idx] = None
    else:
        table[idx, 2] = 1
        x, y = table[idx, :2] - grid_size/2
        rect = patch.Rectangle([x, y], grid_size, grid_size, facecolor=(1, 0, 0, 0.5))
        patch_p = ax.add_patch(rect)
        rectangles[idx] = patch_p

    fig.canvas.draw_idle()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

thetas = np.linspace(0, 2*np.pi, 100)
ax.plot(grid_radius*np.cos(thetas), grid_radius*np.sin(thetas)-8, "r--")
ax.scatter(0, -8, c="r", marker="x") #radar is at (0, -8)

ax.set_aspect('equal')
ax.set_ylabel("North")
ax.set_xlabel("East")

ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
plt.show()
np.save("custom_intensity_grid.npy", table)