import numpy as np

precompute_params = dict()
precompute_params["precompute"] = True
precompute_params["grid_size"] = 5
precompute_params["zrange"] = [-120, 150, -150, 45]


tracker_params = dict()
tracker_params['maximum_velocity'] = 4 #Not used in code, should act as guide to choose init_Pvel
tracker_params['init_Pvel'] = 4 #10**2
tracker_params['P_D'] = 0.7#0.92

# tracker_params['clutter_intensity_file'] = "./clutter_analysis/clutter_intensity_all.npy"
# tracker_params['clutter_intensity_file'] = "./clutter_analysis/custom_intensity_grid.npy"
tracker_params['clutter_intensity_file'] = "./clutter_analysis/custom_clutter_2.npy"
# tracker_params['clutter_intensity_file'] = "./clutter_analysis/clutter_intensity_all_thresholded.npy"

if tracker_params['clutter_intensity_file']:

    intensity = np.load(tracker_params['clutter_intensity_file'])

    def closest_clutter_intensity(pos, low=5e-5, high=5e-2):
        if pos.shape == (2,):
            pos = pos[..., np.newaxis]
        pos_diff = np.linalg.norm(intensity[:, :2] - pos.T[:, np.newaxis, ...], axis=-1)
        min_diff_idx = np.argmin(pos_diff, axis=-1)

        return np.where(intensity[min_diff_idx, 2], high, low)
        # return np.where(intensity[min_diff_idx, 2] > low, intensity[min_diff_idx, 2], low)

    tracker_params['clutter_density'] = closest_clutter_intensity
else:
    tracker_params['clutter_density'] = lambda _: 5e-4


tracker_params['gamma'] = 3.5**2
tracker_params['survival_prob'] = 0.999

tracker_params['birth_intensity_file'] = "./birth_analysis/custom_intensity_grid.npy"

if tracker_params['birth_intensity_file']:

    birth_intensity = np.load(tracker_params['birth_intensity_file'])

    def closest_birth_intensity(pos, low=5e-8, high=5e-3):
        if pos.shape == (2,):
            pos = pos[..., np.newaxis]
        pos_diff = np.linalg.norm(birth_intensity[:, :2] - pos.T[:, np.newaxis, ...], axis=-1)
        min_diff_idx = np.argmin(pos_diff, axis=-1)
        return np.where(birth_intensity[min_diff_idx, 2], high, low)

    tracker_params['birth_intensity'] = closest_birth_intensity
else:
    tracker_params['birth_intensity'] = lambda points: 5e-6 if len(points.shape) == 1 else np.ones(shape=points.shape[1])*5e-6 #1e-7



tracker_params['conf_threshold'] = 0.999
tracker_params['term_threshold'] = 0.3#0.01
tracker_params['visibility_transition_matrix'] = np.array([[0.9, 0.1],[0.52, 0.48]])


measurement_params = dict()
measurement_params['measurement_mapping'] = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0]])
measurement_params['cart_cov'] = 3**2*np.eye(2)#6.6**2*np.eye(2)
measurement_params['range_cov'] = 5**2
measurement_params['bearing_cov'] = ((np.pi/180)*1)**2


process_params = dict()
process_params['init_mode_probs']=np.array([0.8,0.1,0.1])
process_params['cov_CV_low'] = 0.01 #0.1**2
process_params['cov_CV_high'] = 0.05 #1.5**2
process_params['cov_CT'] = 0.02**2
process_params['cov_CV_single'] = 1.5**2

transition_probabilities = [0.99, 0.99, 0.99]

# create transition probability matrix (pi-matrix)
transition_probability_matrix = np.zeros((len(transition_probabilities),len(transition_probabilities)))
for a, transition_probability in enumerate(transition_probabilities):
    transition_probability_matrix[:][a] = (1-transition_probabilities[a])/(len(transition_probabilities)-1)
    transition_probability_matrix[a][a] = transition_probabilities[a]
    try:
        assert(1 - 1e-9 < sum(transition_probability_matrix[:][a]) < 1 + 1e-9)
    except:
        print(transition_probability_matrix[:][a])
        exit()
process_params['pi_matrix'] = transition_probability_matrix
