import numpy as np
import tracking.models as models
from tracking.constructs import State, Track, TrackState
from scipy.linalg import block_diag
import copy
from abc import ABC, abstractmethod

from scipy.stats import norm


class Initiator():
    def __init__(self, initiator, measurement_model, initial_vel_cov, initial_ang_cov, **other_initial_values):
        self.__allowed_initial_states__ = ['kinematic_models']
        self.__allowed_initial_probabilities__ = ['mode_probabilities', 'visibility_probability']
        self.__state_probability_combination__ = {'kinematic_models': 'mode_probabilities'}
        self.__independent_probabilities__ = ['visibility_probability']
        self.__initiator__ = initiator
        self.__initial_vel_cov__ = initial_vel_cov
        self.__initial_ang_cov__ = initial_ang_cov
        self.__measurement_model__ = measurement_model
        self.__dict__.update((k, v) for k, v in other_initial_values.items() if k in self.__allowed_initial_states__)
        self.__dict__.update((k, v) for k, v in other_initial_values.items() if k in self.__allowed_initial_probabilities__)
        self.__index_count__ = 1


class SinglePointInitiator(Initiator):
    def step(self, unused_measurements, **kwargs):
        new_tracks = set()

        # initiate tracks on measurements
        for measurement in unused_measurements:
            # calculate the covariance measurement
            measurement_covariance = self.__measurement_model__.get_measurement_covariance(measurement.value)

            # initialize a track on the measurements
            track = self.__initiate_track__(measurement, measurement_covariance=measurement_covariance)

            # add track to the set of new tracks
            new_tracks.add(track)

            # change the index count, so the next new track gets a different index
            self.__index_count__ += 1
        return new_tracks

    def __initiate_track__(self, measurement, measurement_covariance=None):
        mean = measurement.value
        if measurement_covariance is None:
            R = measurement.covariance
        else:
            R = measurement_covariance

        mean, R = self.__initiator__.get_best_gaussian(mean, R)

        mean = np.hstack((mean, np.zeros(3)))
        covariance = block_diag(R, np.diag([self.__initial_vel_cov__, self.__initial_vel_cov__, self.__initial_ang_cov__]))
        mapping = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 0]])

        mean = mapping.dot(mean)
        covariance = mapping.dot(covariance).dot(mapping.T)
        state = State(mean, covariance)
        state_dict = self.__initialize_states__(state)
        other_states_and_probabilities = dict()
        for k, v in self.__dict__.items():
            if k in self.__allowed_initial_states__ or k in self.__allowed_initial_probabilities__:
                other_states_and_probabilities[k] = v

        # if no kinematic models are specified, a default CV model is chosen.
        if 'kinematic_models' not in other_states_and_probabilities.keys():
            other_states_and_probabilities['kinematic_models'] = models.CVModel(0.1)

        kwargs = {k:v for k, v in self.__dict__.items() if k in self.__independent_probabilities__ or k in self.__allowed_initial_states__}
        new_track = Track(
            measurement.timestamp,
            state_dict,
            self.__index_count__,
            self.__initiator__.initial_existence_probability(measurement.value),
            measurements = {measurement},
            **kwargs
        )
        return new_track

    def __initialize_states__(self, state):
        """
        Initialization of the tree containing the states of the hybrid state.
        """
        states = self.__set_state__(state, None, None, self.__allowed_initial_states__)
        return states

    def __set_state__(self, state, this_discrete_state, root, allowed_initial_states):
        """
        Recursively creates the nodes of the tree. The leaf nodes contain the
        states (or kinematic pdfs), while all parents hold the probabilities of
        their children.
        """
        kwargs = dict()
        for discrete_state in allowed_initial_states:
            if discrete_state in self.__dict__:
                key = self.__state_probability_combination__[discrete_state]
                value = self.__dict__[key]
                kwargs[key] = value
                break

        state_node = TrackState(state.mean, state.covariance, parent = root, name=this_discrete_state, children = None, **kwargs)
        next_allowed_initial_states = copy.deepcopy(allowed_initial_states)
        for discrete_state in allowed_initial_states:
            if discrete_state in self.__dict__:
                next_allowed_initial_states.remove(discrete_state)
                state_node.children = [self.__set_state__(state, d_state, state_node, next_allowed_initial_states) for d_state in self.__dict__[discrete_state]]
                return state_node
            else:
                next_allowed_initial_states.remove(discrete_state)
        return state_node


class StateInitiator(ABC):
    def __init__(self, unknown_influence, clutter_model):
        self.__unknown_influence__ = unknown_influence
        self.__clutter_model__ = clutter_model
    
    @abstractmethod
    def get_best_gaussian(self, mean, sigma):
        pass

    def initial_existence_probability(self, pos):
        u = self.__unknown_influence__.get_unknown_influence(pos) #not compatible with IgnoreUnknown           
        c = self.__clutter_model__.get_clutter_density(pos)
        return float(u/(c + u))


class ApproxStateInitiator(StateInitiator):
    def __init__(self, unknown_influence, clutter_model):
        super().__init__(unknown_influence, clutter_model)
       
    def get_best_gaussian(self, mean, sigma):
        return mean, sigma
    

class PrecomputedStateInitiator(StateInitiator):
    def __init__(self, unknown_influence, clutter_model, precomupted_vals):
        super().__init__(unknown_influence, clutter_model)
        self.__precomp__ = precomupted_vals
    
    def get_best_gaussian(self, mean, sigma):
        n_mean = self.__precomp__.get_expect_at(mean)
        n_sigma = self.__precomp__.get_covar_at(mean)
        return n_mean, n_sigma
    
class OnlineComputeStateInitiator(StateInitiator):
    def __init__(self, unknown_influence, clutter_model, unknown_intensity, num_std=3.5):
        super().__init__(unknown_influence, clutter_model)
        self.unknown_intensity = unknown_intensity
        self.__generate_sampling_points__(num_std)

    def __generate_sampling_points__(self, num_std):
        ##precompute the sample points
        num_points = 5*num_std #num points in each direction within the num_std ellipsis
        #should really also be dependent on the size of the ellipsis (eig_vals), however assuming constant
        #we can use the same discretization for all z's
        self.spacing = 2*num_std/num_points
        x = np.arange(-num_std, num_std, self.spacing)
        x, y = np.meshgrid(x, x)
        x, y = x.ravel(), y.ravel()
        deg2rad = np.pi/180
        points = np.array([[np.cos(45*deg2rad), -np.sin(45*deg2rad)],[np.sin(45*deg2rad), np.cos(45*deg2rad)]])@np.vstack((x, y))
        idx = np.linalg.norm(points, axis=0) <= num_std #remove points outside num_std ellipsis
        self.points = points[:, idx]
        self.normal_values = norm.pdf(self.points[0, :])*norm.pdf(self.points[1, :]) #normalized sample points values
    
    def get_best_gaussian(self, mean, sigma):
        sigma = (sigma + sigma.T)/2 #ensure symmetry, shouldn't really be necessary
        eig_vals, eig_vecs = np.linalg.eig(sigma)
        eig_root = np.diag(np.sqrt(eig_vals))
        tpoints = eig_vecs@eig_root@self.points + mean[..., np.newaxis]

        us = self.unknown_intensity(tpoints)

        integral = self.spacing**2*us@self.normal_values
        nmean = self.spacing**2*(us*tpoints)@self.normal_values/integral
        err = tpoints - nmean[..., np.newaxis]
        err_outer = np.einsum("ba,ca -> abc", err, err)
        ue = np.einsum("i,ijk -> ijk", us, err_outer)
        nsigma = self.spacing**2*np.einsum("ijk, i -> jk", ue, self.normal_values)/integral

        return nmean, nsigma