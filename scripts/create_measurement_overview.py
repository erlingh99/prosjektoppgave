import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
from os.path import basename


parser = argparse.ArgumentParser(
    prog="measurement_vis",
    description="Takes a .npy file of measurements and creates a plot."
)

parser.add_argument("input_file")
file = parser.parse_args().input_file

data = np.load(file, allow_pickle=True).item()
east = np.array(data["y"])
north = np.array(data["x"])
ts = np.array(data["timestamp"])
idx = north < 45
east = east[idx]
north = north[idx]
ts = ts[idx]

plt.figure(figsize=(10,10))
max_age = ts[-1]
grayscale_values = [1.0 - (age / max_age) for age in ts]

basemap = "ravnkloa_radar_base_map"

with open(f"./{basemap}.yaml", "r") as f:
    doc = yaml.safe_load(f)

map = np.fromfile(f"./{basemap}.bin", dtype=bool)
basemap = map.reshape((doc["map_height"], doc["map_width"])).T
offset = (doc["origin_x_coordinate"], doc["origin_y_coordinate"])

plt.imshow(basemap, origin="lower",   
            extent=[offset[1],
                    offset[1] + basemap.shape[1],
                    offset[0],
                    offset[0] + basemap.shape[0]],
            cmap="BuGn",
            vmin=-1,#to get a bit less blue water
            vmax=2) #to get a bit less green land

#remvoe y over 40
plt.scatter(east, north, c=grayscale_values, cmap="Greys_r")
plt.scatter(0,0, c='red', marker='o',linewidths=1, label='Origin (Fosenkaia NED)')
plt.plot([0, 10], [0, 0], "b")
plt.plot([0, 0], [0, 10], "r")
plt.scatter(0, -8, marker="o", color="b", linewidths=5, label="radar")
plt.xlim(-150, 150)
plt.ylim(-200, 100)
plt.legend()
plt.show()
plt.savefig("./scenarios/plots/" + basename(file)[:-4]+".png")