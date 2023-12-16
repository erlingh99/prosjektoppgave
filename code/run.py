from tracking import filters, models, initiators, terminators, managers, associators, trackers, precomputer, unknown_integral
from parameters import tracker_params, measurement_params, process_params, precompute_params

import import_data
import plotting
import numpy as np
from matplotlib.animation import FFMpegWriter
import argparse

import time

import matplotlib.pyplot as plt

import tqdm
from os.path import join, basename, exists
from os import mkdir

def setup_manager():
    if IMM_off:
        kinematic_models = [models.CVModel(process_params['cov_CV_high'])]
        pi_matrix = np.array([[1]])
        init_mode_probs = np.array([1])
    else:
        kinematic_models = [models.CVModel(process_params['cov_CV_low']),models.CTModel(process_params['cov_CV_low'],process_params['cov_CT']),models.CVModel(process_params['cov_CV_high'])]
        pi_matrix = process_params['pi_matrix']
        init_mode_probs = process_params['init_mode_probs']

    measurement_model = models.CombinedMeasurementModel(
        measurement_mapping = measurement_params['measurement_mapping'],
        cartesian_covariance = measurement_params['cart_cov'],
        range_covariance = measurement_params['range_cov'],
        bearing_covariance = measurement_params['bearing_cov'])
    
    clutter_model = models.ConstantClutterModel(tracker_params['clutter_density'])

    if precompute_params["precompute"]:
        precomp = precomputer.Precomputer(tracker_params['birth_intensity'],
                                            measurement_model,
                                            zrange=precompute_params["zrange"],
                                            grid_size=precompute_params["grid_size"],
                                            num_std=3.5)
        
        unknown_influence_model = unknown_integral.PrecomputedTargetInfluence(tracker_params['birth_intensity'], tracker_params["P_D"], precomp)
        initiator = initiators.PrecomputedStateInitiator(unknown_influence_model, clutter_model, precomp)

        # unknown_influence_model = unknown_integral.ApproxTargetInfluence(tracker_params['birth_intensity'], tracker_params["P_D"])
        # initiator = initiators.ApproxStateInitiator(unknown_influence_model, clutter_model)  
    else:
        #slow
        # unknown_influence_model = unknown_integral.OnlineComputedTargetInfluence(tracker_params['birth_intensity'],
        #                                                                           tracker_params["P_D"],
        #                                                                           measurement_model)
        # initiator = initiators.OnlineComputeStateInitiator(unknown_influence_model,
        #                                                     clutter_model,
        #                                                     tracker_params['birth_intensity'])  


        # unknown_influence_model = unknown_integral.ApproxTargetInfluence(tracker_params['birth_intensity'], tracker_params["P_D"])
        unknown_influence_model = unknown_integral.GaussianSmoothTargetInfluence(tracker_params['birth_intensity'], tracker_params["P_D"])
        initiator = initiators.ApproxStateInitiator(unknown_influence_model, clutter_model)  


    filter = filters.IMMFilter(
        measurement_model = measurement_model,
        mode_transition_matrix = pi_matrix)

    data_associator = associators.MurtyDataAssociator(
        n_measurements_murty = 4,
        n_tracks_murty = 2,
        n_hypotheses_murty = 8)

    tracker = trackers.VIMMJIPDATracker(
        filter,
        clutter_model,
        unknown_influence_model,
        data_associator,
        survival_probability=tracker_params['survival_prob'],
        visibility_transition_matrix = tracker_params['visibility_transition_matrix'],
        detection_probability=tracker_params['P_D'],
        gamma=tracker_params['gamma'],
        single_target=single_target,
        visibility_off=visibility_off)

    track_initiation = initiators.SinglePointInitiator(
        initiator,
        measurement_model,
        tracker_params['init_Pvel'],
        tracker_params['init_Pvel_ang'],
        mode_probabilities = init_mode_probs,
        kinematic_models = kinematic_models,
        visibility_probability = 0.9)

    track_terminator = terminators.Terminator(
        tracker_params['term_threshold'],
        max_steps_without_measurements = 5,
        fusion_significance_level = 0.01)

    track_manager = managers.Manager(tracker, track_initiation, track_terminator, tracker_params['conf_threshold'])
    return track_manager



if __name__ == '__main__':
    """
    All tracker parameters are imported from parameters.py, and can be changed
    there.
    """

    parser = argparse.ArgumentParser(
        prog="VIMMJIPDA_tracker",
        description="Takes a .npy file of measurements and creates tracks. Parameters are loaded from parameters.py"
    )

    parser.add_argument("input_file")
    parser.add_argument("-d", "--output_dir")
    parser.add_argument("-a", "--analyze_birth", action='store_true')
    parser.add_argument("-p", "--create_plot", action='store_true')
    parser.add_argument("-m", "--create_movie", action='store_true')
    parser.add_argument("-b", "--base_map_file")
    args = parser.parse_args()

    file = args.input_file
    dir = args.output_dir
    if not dir:
        dir = "./results"
    if not exists(dir):
        mkdir(dir)

    analyze_birth = args.analyze_birth
    make_movie = args.create_movie
    make_plot = args.create_plot
    basemap = args.base_map_file


    joyride = False
    final_dem = False

    # select the part of the data sets to import
    if joyride:
        t_max = 10000
        t_min = 0

    if final_dem:
        t_max = 1300
        t_min = 900

    # turn off tracker functionality
    IMM_off = False
    single_target = False
    visibility_off = True

    # import data
    if joyride:
        measurements, ownship, ground_truth, timestamps = import_data.joyride(t_min=t_min, t_max=t_max)
    elif final_dem:
        measurements, ownship, ground_truth, timestamps = import_data.final_dem(t_min=t_min, t_max=t_max) #ground_truth here refers to the Gunnerus AIS data
    elif file:
        measurements, ownship, ground_truth, timestamps = import_data.from_radar(file) #ground_truth is null
    else:
        raise ValueError("No data chosen")


    # define tracker evironment
    manager = setup_manager()

    now = time.time()
    # run tracker
    for k, (measurement_set, timestamp, ownship_pos) in enumerate(zip(measurements, timestamps, *ownship.values())):
        print(f'Timestep {k}:')
        manager.step(measurement_set, float(timestamp), ownship=ownship_pos)
        print(f'Active tracks: {np.sort([track.index for track in manager.tracks])}\n')

    print(f"Used tracker time: {(time.time()-now):.1f}s")
            
    # plotting
    plot = plotting.ScenarioPlot(
        measurement_marker_size=3,
        track_marker_size=5,
        add_covariance_ellipses=True,
        add_validation_gates=False,
        add_track_indexes=False,
        gamma=3.5,
        basemap = basemap,
        plot_lims = [-200, 50, -150, 100]
    )


    ### save starting points of tracks
    if analyze_birth:
        path = join(dir, "birthstats.txt")

        birth_stats = []
        for v in manager.track_history.values():
            mean, _ = v[0].posterior #extract the birth pos of each track
            x = mean[0]
            y = mean[2]
            t = v[0].timestamp
            birth_stats.append((x, y, t))

        with open(path, "a") as f:
            f.writelines(str(l)+"\n" for l in birth_stats)

        print(f"Birthstats for {file} saved to {path}.")

    if make_plot:
        print("Creating plot")
        save_path = join(dir, f"{basename(file).split('.')[0]}.png")
        save_path = None
        plot.create_fig(save_path, measurements, manager.track_history, 
                        ownship, timestamps, ground_truth)
        
        for idx, t in manager.track_history.items():
            times = np.array([tt.timestamp for tt in t])
            if len(times) < 20:
                continue
            probs = np.array([tt.mode_probabilities for tt in t])
            plt.plot(times, probs[:, 0], label="CV_low")
            plt.plot(times, probs[:, 1], label="CT")
            plt.plot(times, probs[:, 2], label="CV_high")
            plt.title(f"track {idx}")
            plt.legend()
            plt.ylim([0, 1])
            plt.show()
        
    
    if make_movie:
        writer = FFMpegWriter(fps=5)

        save_path = join(dir, f"{basename(file).split('.')[0]}.mp4")
        with writer.saving(plot.fig, save_path, 100):
            print("Creating movie")
            for i in tqdm.tqdm(range(2, len(timestamps))):
                m = measurements[:i]
                t = timestamps[:i]
                o = {k: v[:i] for k,v in ownship.items()}            
                mth = {k: [vv for vv in v if vv.timestamp <= timestamps[i]] for k, v in manager.track_history.items()}

                plot.ax.cla()
                if ground_truth:
                    gt = {k: v[:i] for k,v in ground_truth.items()}
                    plot.create(m, mth, o, t, gt)
                else:
                    plot.create(m, mth, o, t)

                writer.grab_frame()
