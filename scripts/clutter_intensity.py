import matplotlib.pyplot as plt
import numpy as np
import glob
import yaml

measurements = []

files = glob.glob("./data/npy/*.npy")
for file in files:
    data = np.load(file, allow_pickle=True).item()
    meas = list(zip(data["y"], data["x"]))
    meas = [m for m in meas if m[1] < 45]
    measurements.extend(meas)

measurements = np.array(measurements)
x = measurements[:,0]
y = measurements[:,1]


with open("./ravnkloa_radar_base_map.yaml", "r") as f:
    doc = yaml.safe_load(f)

map = np.fromfile("./ravnkloa_radar_base_map.bin", dtype=bool).reshape((doc["map_height"], doc["map_width"])).T

plt.imshow(map, origin="lower",   
            extent=[doc["origin_y_coordinate"],
                    doc["origin_y_coordinate"] + doc["map_height"],
                    doc["origin_x_coordinate"],
                    doc["origin_x_coordinate"] + doc["map_width"]],
            cmap="BuGn",
            vmin=-1,#to get a bit less blue water
            vmax=2)#to get a bit less green land

plt.scatter(x, y)
plt.xlim(-200, 200)
plt.ylim(-200, 200)
plt.show()


num_scans = 63_337 #estimated
grid_size = 10 #meters
grid_radius = 150 #150 meters in each direction from radar, so 300x300 meter
grid_shape = (2*grid_radius)//grid_size
grid = np.zeros(shape=(grid_shape, grid_shape))

hist, xedges, yedges = np.histogram2d(x, y, bins=grid_shape, range=[[-grid_radius, grid_radius], [-grid_radius, grid_radius]])



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + grid_size//2, yedges[:-1] + grid_size//2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
#construct bar size
dx = dy = grid_size//2 * np.ones_like(zpos)
#intensity = count / num scans / cell size 
dz = hist.ravel()/num_scans/(grid_size**2)
# np.save("./clutter_analysis/clutter_intensity_all.npy", list(zip(xpos, ypos, dz)))
print(np.average(np.sort(dz)[-20:]))

map = map.astype(float)
mapsize = 400
#discard areas outside of mapsize/2 from center
cx = int(-doc["origin_x_coordinate"])
cy = int(-doc["origin_y_coordinate"])
map = map[cx-mapsize//2:cx+mapsize//2, cy-mapsize//2:cy+mapsize//2]

# stride arg determines image quality 
stride = 1
stepX, stepY = mapsize / (map.shape[1]/stride), mapsize / (map.shape[0]/stride)

X1 = np.arange(-mapsize/2, mapsize/2, stepX)
Y1 = np.arange(-mapsize/2, mapsize/2, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
ax.scatter(X1, Y1, np.zeros_like(X1), c=map[::stride,::stride], cmap="BuGn", vmin=-1, vmax=2)


ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.plot([0, 15], [0, 0], [0, 0], "r", linewidth=1)
ax.plot([0, 0], [0, 15], [0, 0], "g", linewidth=1)
ax.scatter(0, -8, 0, c="k", linewidths=1)

ax.view_init(elev=35, azim=-90)

plt.ylabel("North")
plt.xlabel("East")
plt.show()
