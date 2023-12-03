import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import yaml

from matplotlib.cm import get_cmap
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon
from descartes import PolygonPatch

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.figsize'] = 7.16666, 7.166666


class ScenarioPlot(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, 
                measurement_marker_size=3,
                track_marker_size=5, 
                add_covariance_ellipses=False, 
                add_validation_gates=False, add_track_indexes=False, 
                gamma=3.5, 
                basemap=None,
                plot_lims=None):
        self.track_marker_size = track_marker_size
        self.measurement_marker_size = measurement_marker_size
        self.add_track_indexes = add_track_indexes
        self.add_validation_gates = add_validation_gates
        self.add_covariance_ellipses = add_covariance_ellipses
        self.gamma = gamma
        self.plot_lims = plot_lims
        self.fig, self.ax = plt.subplots()

        self.basemap = None
        if basemap:
            with open(f"./{basemap}.yaml", "r") as f:
                doc = yaml.safe_load(f)

            map = np.fromfile(f"./{basemap}.bin", dtype=bool)
            self.basemap = map.reshape((doc["map_height"], doc["map_width"])).T
            self.offset = (doc["origin_x_coordinate"], doc["origin_y_coordinate"])
            


    def create(self, measurements, track_history, ownship, timestamps, ground_truth=None, plot_lims=None):
        if self.basemap is not None:
            self.ax.imshow(self.basemap, origin="lower",   
                            extent = [self.offset[1],
                                    self.offset[1] + self.basemap.shape[1],
                                    self.offset[0],
                                    self.offset[0] + self.basemap.shape[0]],
                                cmap="BuGn",
                                vmin=-1,#to get a bit less blue "water"
                                vmax=2)
            
        plot_track_pos(ownship, self.ax, marker_size=self.track_marker_size, color='gray')
        plot_measurements(measurements, self.ax, timestamps, marker_size=self.measurement_marker_size)
        if ground_truth:
            plot_track_pos(ground_truth, self.ax, color='k', marker_size=self.track_marker_size)

        plot_track_pos(
            track_history,
            self.ax,
            add_index=self.add_track_indexes,
            add_covariance_ellipses=self.add_covariance_ellipses,
            add_validation_gates=self.add_validation_gates,
            gamma=self.gamma)

        if self.plot_lims is None:
            N_min, N_max, E_min, E_max = find_track_limits(track_history)
        else:
            N_min, N_max, E_min, E_max = self.plot_lims

        self.ax.set_xlim(E_min, E_max)
        self.ax.set_ylim(N_min, N_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')

    def create_fig(self, savepath, measurements, track_history, ownship, timestamps, ground_truth=None):
        self.create(measurements, track_history, ownship, timestamps, ground_truth)
        # self.fig.savefig(savepath, dpi=600, format=savepath.split(".")[-1])

        plt.show()
        

def plot_measurements(measurements_all, ax, timestamps, marker_size=5):
    cmap = get_cmap('Greys')
    measurements_all = dict((i, set(measurements)) for i, measurements in enumerate(measurements_all))

    timestamps = np.asarray(timestamps)
    interval = (timestamps-timestamps[0]+timestamps[-1]/5)/(timestamps[-1]-timestamps[0]+timestamps[-1]/5)
    for index, measurement_set in measurements_all.items():
        color = cmap(interval[index].squeeze())
        for measurement in measurement_set:
            ax.plot(measurement.value[0], measurement.value[1], marker='o', color=color, markersize=marker_size)


def plot_track_pos(track_history, 
                   ax, 
                   add_index=False, 
                   add_covariance_ellipses=False, 
                   add_validation_gates=False, 
                   gamma=3.5, 
                   lw=1, 
                   ls='-', 
                   marker_size = 5, 
                   color=None,):
    color_idx = 0
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#ff0000', '#00ffff', "#01EA3E", "#2355E8", "#C0E80C", "#EB72EA", "#E8A800"]

    for index, trajectory in track_history.items():
        if len(trajectory) == 0:
            continue

        positions = np.array([track.posterior[0] for track in trajectory])

        if color is not None:
            selected_color = color
        else:
            selected_color = colors[color_idx%len(colors)]
            color_idx += 1

        line, = ax.plot(positions[:,0], positions[:,2], color=selected_color, lw=lw,ls=ls)
        last_position, = ax.plot(positions[-1,0], positions[-1,2], 'o', color=selected_color, markersize=marker_size)

        if add_covariance_ellipses:
            edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0)
            facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
            for track in trajectory:
                covariance_ellipse = get_ellipse(track.posterior[0][0:3:2], track.posterior[1][0:3:2,0:3:2])
                ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))

        if add_validation_gates:
            edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.5)
            facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
            for track in trajectory[1:]: # the first track estimate has no validation gate
                validation_gate = get_validation_gate(track, gamma=gamma)
                ax.add_patch(PolygonPatch(validation_gate, facecolor = facecolor, edgecolor = edgecolor))

        if add_index:
            ax.text(positions[-1,0], positions[-1,2]-5, str(index), color='black')


def get_validation_gate(state, gamma):
        for j, predicted_measurement in enumerate(state.predicted_measurements.leaves):
            validation_gate = get_ellipse(predicted_measurement.mean, predicted_measurement.covariance, gamma)
            if j == 0:
                union = validation_gate
            else:
                if validation_gate.intersection(union).is_empty:
                    warnings.warn(f'The validation gates for the kinematic pdfs for track {state.index} at time {state.timestamp} are disjoint. The displayed validation gate will be displayed incorrectly.')
                    pass
                else:
                    union = validation_gate.union(union)
        return union


def get_ellipse(center, Sigma, gamma=1):
    """
    Returns an ellipse. For a covariance ellipse, gamma is the square of the
    number of standard deviations from the mean.
    Method from https://cookierobotics.com/007/.
    """
    lambda_, _ = np.linalg.eig(Sigma)
    lambda_root = np.sqrt(lambda_)
    width = lambda_root[0]*np.sqrt(gamma)
    height = lambda_root[1]*np.sqrt(gamma)
    rotation = np.rad2deg(np.arctan2(lambda_[0]-Sigma[0,0], Sigma[0,1]))
    circ = Point(center).buffer(1)
    non_rotated_ellipse = affinity.scale(circ, width, height)
    ellipse = affinity.rotate(non_rotated_ellipse, rotation)
    edge = np.array(ellipse.exterior.coords.xy)
    return Polygon(edge.T)

def find_track_limits(track_history, extra_spacing=50):
    N_min, N_max, E_min, E_max = np.inf, -np.inf, np.inf, -np.inf
    for track_id, trajectory in track_history.items():
        for track in trajectory:
            mean = track.posterior[0]
            if mean[2] < N_min:
                N_min = mean[2]
            if mean[2] > N_max:
                N_max = mean[2]
            if mean[0] < E_min:
                E_min = mean[0]
            if mean[0] > E_max:
                E_max = mean[0]
    N_min -= extra_spacing
    N_max += extra_spacing
    E_min -= extra_spacing
    E_max += extra_spacing

    N_min = min(N_min, -20)
    N_max = max(N_max, 20)
    E_min = min(E_min, -20)
    E_max = max(E_max, 20)

    return N_min, N_max, E_min, E_max
