#!/bin/bash
#run from the VIMMJIPDA folder
#./track_birth.bash path_to_npy_files

source ./bin/activate

files=$(find $1 -type f -name "*.npy")
n=$(echo "$files" | wc -l)
echo "$n files from $1 will be analyzed."

outdir=./VIMMJIPDA/birth_analysis 

echo "analyzing..."

i=1
for file in $files
do
    echo -ne "\rtracking file $i/$n"
    ./bin/python ./VIMMJIPDA/code/run.py $file -d $outdir -a > /dev/null
    ((i++))
done

echo -e "\rdone"
./bin/python visualize_birth.py