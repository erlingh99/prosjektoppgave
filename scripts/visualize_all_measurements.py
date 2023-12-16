import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob
"""
Plots the map and measurements in the Fosenkaia NED-frame (which both map and measurements are in).
Also plots the frame, and the pos of the Navico radar, with a 150m radius circle around the radar
"""

with open("./ravnkloa_radar_base_map.yaml", "r") as f:
    doc = yaml.safe_load(f)

map = np.fromfile("./ravnkloa_radar_base_map.bin", dtype=bool).reshape((doc["map_height"], doc["map_width"])).T


x = []
y = []

files = glob.glob("./data/npy/*.npy")
for f in files:
    meas = np.load(f, allow_pickle=True).item()
    x.extend(meas["y"])
    y.extend(meas["x"])

x = np.array(x)    
y = np.array(y)  

idx = y < 40
x = x[idx]
y = y[idx]


plt.imshow(map, origin="lower",   
           extent = [doc["origin_y_coordinate"],
                     doc["origin_y_coordinate"] + doc["map_height"],
                     doc["origin_x_coordinate"],
                     doc["origin_x_coordinate"] + doc["map_width"]],
            cmap="BuGn",
            vmin=-1,#to get a bit less blue "water"
            vmax=2)
plt.scatter(x, y, marker="x", color="r")
plt.plot([0, 10], [0, 0], "g")
plt.plot([0, 0], [0, 10], "r")
plt.scatter(0, -8, marker="o", color="g")
plt.plot([150*np.cos(t) for t in np.linspace(0, 2*3.14, 100)], [150*np.sin(t) - 8 for t in np.linspace(0, 2*3.14, 100)], "g")
plt.xlim(-200, 200)
plt.ylim(-200, 200)
plt.gca().set_aspect('equal')
plt.show()