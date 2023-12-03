#!/bin/bash

#./record_markers.bash path_to_bag_dir
#this file starts the aferry pipeline and plays the raw-data bags one by one
#it records all markers generated in the pipeline and extracts the marker positions to .npy files
#huge reduction in size between bag and npy
#the npy files can then be tracked by the VIMMJIPDA, see track_birth.bash

files=$(find $1 -type f -name "*.bag")

echo "Creating marker bags from these files:"
echo "$files"

source /opt/ros/noetic/setup.bash
source ~/prosjektoppgave/catkin/devel/setup.bash

roscore &
echo "waiting for ros to start"
sleep 5
roslaunch ros_af_program_start start_program.launch &

for file in $files
do 
  outname="${file::-4}-markers.bag"
  
  rosbag record /radar/detector/cluster_vis/markers -O $outname __name:=marker_bag &
  rosbag play $file -r 100
  rosnode kill /marker_bag

  python3.8 ~/prosjektoppgave/VIMMJIPDA/VIMMJIPDA/scripts/bag_to_npy.py -d ~/prosjektoppgave/VIMMJIPDA/VIMMJIPDA/data -p $outname

  rm $outname
done
