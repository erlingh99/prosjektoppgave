import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

class UnknownTargetInfluence(ABC):
    """
    A model for the unknown target influence in calculating assocation probabilities and new target.
    Serves to approximate the integral in a few different ways.
    """
    def __init__(self, unknown_target_intensity, P_d):
        self.__unknown_target_intensity__ = unknown_target_intensity
        self.__P_d__ = P_d

    @abstractmethod
    def get_unknown_influence(self, pos):
        pass

class IgnoreUnknownTargetInfluence(UnknownTargetInfluence):
    def __init__(self, unknown_target_intensity, P_d):
        super().__init__(unknown_target_intensity, P_d)
    
    def get_unknown_influence(self, pos):
        return 0

class ApproxTargetInfluence(UnknownTargetInfluence):
    def __init__(self, unknown_target_intensity, P_d):
        super().__init__(unknown_target_intensity, P_d)
    
    def get_unknown_influence(self, pos):
        return self.__unknown_target_intensity__(pos)*self.__P_d__
    
class PrecomputedTargetInfluence(UnknownTargetInfluence):
    def __init__(self, unknown_target_intensity, P_d, precomputed):
        super().__init__(unknown_target_intensity, P_d)
        self.__precomputed_vals__ = precomputed

    def get_unknown_influence(self, pos):
        return self.__precomputed_vals__.get_integral_at(pos)*self.__P_d__
    
class OnlineComputedTargetInfluence(UnknownTargetInfluence):
    def __init__(self, unknown_target_intensity, P_d, measurement_model, num_std=3.5):
        super().__init__(unknown_target_intensity, P_d)
        self.__measurement_model__ = measurement_model

        self.prev_pos = None
        self.prev_val = None
        
        self.__generate_sampling_points__(num_std)

    def __generate_sampling_points__(self, num_std):
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

    def get_unknown_influence(self, mean):
        if np.all(mean == self.prev_pos):#often called on same values many times
            return self.prev_val

        sigma = self.__measurement_model__.get_measurement_covariance(mean)
        sigma = (sigma + sigma.T)/2 #ensure symmetry, shouldn't really be necessary
        eig_vals, eig_vecs = np.linalg.eig(sigma)
        eig_root = np.diag(np.sqrt(eig_vals))
        tpoints = eig_vecs@eig_root@self.points + mean[..., np.newaxis]

        us = self.__unknown_target_intensity__(tpoints)

        u = self.__P_d__*self.spacing**2*us@self.normal_values

        self.prev_pos = mean
        self.prev_val = u

        return u