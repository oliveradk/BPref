#!/bin/sh
. ./tests/run_PEBBLE_metaworld_grasp.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld succeeded"
else
    echo "PEBBLE metaworld failed"
    exit 1
fi

. ./tests/run_PEBBLE_dmc.sh
if [ $? -eq 0 ]; then 
    echo "PEBBLE dmc succeeded" 
else
    echo "PEBBLE dmc failed" 
    exit 1
fi 

. ./tests/run_sac_metaworld.sh
if [ $? -eq 0 ]; then 
    echo "SAC metaworld succeeded" 
else
    echo "SAC metaworld failed" 
    exit 1
fi 

. ./tests/run_sac_dmc.sh
if [ $? -eq 0 ]; then 
    echo "SAC dmc succeeded" 
else
    echo "SAC dmc failed" 
    exit 1
fi