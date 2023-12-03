import numpy as np
import rosbag
import matplotlib.pyplot as plt

from datetime import datetime
from os.path import join, dirname, exists
from os import mkdir
import argparse
import yaml

# I want the BagReader class to save the x, y and timestamp in a directory
class BagReader:
    def __init__(self, bag_path):
        self.bag = rosbag.Bag(bag_path)
        self.topic = '/radar/detector/cluster_vis/markers'
        self.data = {'x': [], 'y': [], 'timestamp': []}

    def __repr__(self) -> str:
        return f"{self.data['x']}, {self.data['y']}, {self.data['timestamp']}"

    def read_bag(self):
        warned = False
        frame = ""


        for topic, msg, t in self.bag.read_messages(topics=[self.topic]):
            if msg.markers:
                for marker in msg.markers:
                    if marker.header.frame_id != frame:
                        frame = marker.header.frame_id
                        print(f"Marker in frame {frame}")
                    

                    if np.abs(marker.pose.position.x) < 1 and np.abs(marker.pose.position.y) < 1: #skip bug points at radar position
                        if not warned:
                            print("The dataset contains points at the sensor location, ignoring those points...")
                            warned = True
                        continue

                    self.data['x'].append(marker.pose.position.x)
                    self.data['y'].append(marker.pose.position.y)
                    # I want the first timestamp to be 0 and the rest to be relative to the first timestamp
                    if not self.data['timestamp']:
                        self.record_time = marker.header.stamp
                        self.first_timestamp = self.record_time.secs
                        self.data['timestamp'].append(0)
                    else:
                        self.data['timestamp'].append(marker.header.stamp.secs - self.first_timestamp)

    def close_bag(self):
        self.bag.close()

    # I want to plot the x and y values with a colormap based on the timestamp
    def plot(self, file):
        plt.figure(figsize=(10,10))
        max_age = self.data['timestamp'][-1]
        grayscale_values = [1.0 - (age / max_age) for age in self.data['timestamp']]
        
        basemap = "ravnkloa_radar_base_map"

        with open(f"./{basemap}.yaml", "r") as f:
            doc = yaml.safe_load(f)

        map = np.fromfile(f"../{basemap}.bin", dtype=bool)
        basemap = map.reshape((doc["map_height"], doc["map_width"])).T
        offset = (doc["origin_x_coordinate"], doc["origin_y_coordinate"])

        plt.imshow(basemap, origin="lower",   
                    extent = [offset[1],
                              offset[1] + basemap.shape[1],
                              offset[0],
                              offset[0] + basemap.shape[0]],
                    cmap="BuGn",
                    vmin=-1,#to get a bit less blue water
                    vmax=2) #to get a bit less green land
        
        plt.scatter(self.data['y'], self.data['x'], c=grayscale_values, cmap="Greys")
        # I want to scatter the radar position at (0,0) and the first and last timestamp with different colors
        plt.scatter(0,0, c='red', marker='o',linewidths=1, label='Origin (Fosenkaia NED)')
        plt.plot([0, 10], [0, 0], "b")
        plt.plot([0, 0], [0, 10], "r")
        plt.scatter(0, -8, marker="o", color="b", linewidths=5, label="radar")
        plt.scatter(self.data['y'][0], self.data['x'][0], c='yellow', marker='o',linewidths=5,label='First timestamp')
        plt.scatter(self.data['y'][-1], self.data['x'][-1], c='orange', marker='o',linewidths=5,label='Last timestamp')
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        plt.legend()
        plt.savefig(file)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog="Bag_to_npy",
        description="Converts bag with the /radar/detector/cluster_vis/markers topic to a .npy file of x,y-coordinates and timestamps of measurements."
    )

    parser.add_argument("input_file")
    parser.add_argument("-d", "--output_dir")
    parser.add_argument("-p", "--create_plot", action='store_true')
    args = parser.parse_args()

    path = args.input_file   
    output_dir = dirname(path)
    if args.output_dir:
        output_dir = args.output_dir

    print(f"\nReading bag at '{path}' and saving to '{output_dir}'.")
    bag = BagReader(path)
    bag.read_bag()
    bag.close_bag()


    time = datetime.utcfromtimestamp(bag.record_time.secs + 2*60*60).strftime("%Y-%m-%d-%H-%M-%S") #add 2 hours to get local time

    out_npy = join(output_dir, "npy", time) + ".npy"
    out_plot = join(output_dir, "plot", time) + ".png"

    if not exists(dirname(out_npy)):
        mkdir(dirname(out_npy))
    if not exists(dirname(out_plot)) and args.create_plot:
        mkdir(dirname(out_plot))


    np.save(out_npy, bag.data)
    print(f"npy saved at {out_npy}")
    if args.create_plot:
        bag.plot(out_plot)
        print(f"Plot saved at {out_plot}")
