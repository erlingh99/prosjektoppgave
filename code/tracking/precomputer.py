import numpy as np
from scipy.stats import norm
import time

class Precomputer:
    def __init__(self, unknown_target_intensity, measurement_model, zrange, grid_size, num_std = 3.5):
        self.__unknown_target_intensity__ = unknown_target_intensity
        self.__precompute_integral__(measurement_model, zrange, grid_size, num_std)

    def __precompute_integral__(self, measurement_model, zrange, grid_size, num_std):
        measurement_model.__ownship_position__ = np.array([0,-8]) #cheat, but since we know the radar doesn't move we can precompute
        #this is the easiest way to do this compatible with the measurement model, which is update with the ownship_pos
        #each timestep. However we know it doesnt move, and is always at 0, -8
        start = time.time()
        print(f"Precomputing integral over [{zrange[0]}, {zrange[1]}]x[{zrange[2]}, {zrange[3]}] with grid size {grid_size}x{grid_size}...",end="",flush=True)

        zs = self.__generate_z_sampling_points__(zrange, grid_size)
        points, spacing = self.__generate_integral_sampling_points__(num_std)

        normal_values = norm.pdf(points[0, :])*norm.pdf(points[1, :]) #normalized sample points values

        self.u_int = np.zeros((len(zs), 3)) #2 coordinates + 1 value
        self.expect = np.zeros((len(zs), 4)) #2 coordinates + mean (2D)
        self.covar = np.zeros((len(zs), 6)) #2 + 2x2 
        for i, z in enumerate(zs):
            sigma = measurement_model.get_measurement_covariance(z)
            sigma = (sigma + sigma.T)/2 #ensure symmetry, shouldn't really be necessary
            eig_vals, eig_vecs = np.linalg.eig(sigma)
            eig_root = np.diag(np.sqrt(eig_vals))
            tpoints = eig_vecs@eig_root@points + z[..., np.newaxis]
            us = self.__unknown_target_intensity__(tpoints)
            
            #compute the interal
            integral = spacing**2*us@normal_values
            #compute the expectation
            expect = spacing**2*(us*tpoints)@normal_values/integral
            #compute the variance
            err = tpoints - expect[..., np.newaxis]
            err_outer = np.einsum("ba,ca -> abc", err, err)
            ue = np.einsum("i,ijk -> ijk", us, err_outer)
            var = spacing**2*np.einsum("ijk, i -> jk", ue, normal_values)/integral
            
            self.u_int[i] = np.array([*z, integral])
            self.expect[i] = np.array([*z, *expect])
            self.covar[i] = np.array([*z, *var.ravel()])

        print(f"Done. Elapsed time: {(time.time() - start):.1f}s")

    def __generate_z_sampling_points__(self, zrange, grid_size):
        z_1d_x = np.arange(zrange[0], zrange[1], grid_size) + grid_size/2
        z_1d_y = np.arange(zrange[2], zrange[3], grid_size) + grid_size/2
        z_1, z_2 = np.meshgrid(z_1d_x, z_1d_y)
        z_1, z_2 = z_1.ravel(), z_2.ravel()
        zs = np.vstack((z_1, z_2)).T
        return zs

    def __generate_integral_sampling_points__(self, num_std):
        num_points = 5*num_std #num points in each direction within the num_std ellipsis
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
        return points, spacing

    def get_integral_at(self, pos, low=1e-2, k=4):
        pos_diff = np.linalg.norm(self.u_int[:, :2] - pos, axis=-1)
        min_idx = np.argpartition(pos_diff, k, axis=-1)[:k]

        if min(pos_diff[min_idx]) < low: #avoid inf weights
            idx = np.argmin(pos_diff[min_idx])
            return self.u_int[min_idx[idx], 2]
        
        weights = 1/pos_diff[min_idx]
        integrals = self.u_int[min_idx, 2]
        return np.average(integrals, weights=weights)
    
    def get_expect_at(self, pos, low=1e-2, k=4):
        pos_diff = np.linalg.norm(self.expect[:, :2] - pos, axis=-1)
        min_idx = np.argpartition(pos_diff, k, axis=-1)[:k]

        if min(pos_diff[min_idx]) < low:
            idx = np.argmin(pos_diff[min_idx])
            return self.expect[min_idx[idx], 2:].reshape((2,))
        
        weights = 1/pos_diff[min_idx]
        expects = self.expect[min_idx, 2:].reshape(-1, 2)
        return np.average(expects, axis=0,  weights=weights)
    
    def get_covar_at(self, pos, low=1e-2, k=4):
        pos_diff = np.linalg.norm(self.expect[:, :2] - pos, axis=-1)
        min_idx = np.argpartition(pos_diff, k, axis=-1)[:k]

        if min(pos_diff[min_idx]) < low:
            idx = np.argmin(pos_diff[min_idx])
            return self.covar[min_idx[idx], 2:].reshape((2, 2))

        weights = 1/pos_diff[min_idx]
        covars = self.covar[min_idx, 2:].reshape((-1, 2, 2))
        return np.average(covars, axis=0, weights=weights)