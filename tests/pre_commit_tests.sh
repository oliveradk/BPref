#!/bin/sh
. ./tests/run_PEBBLE_metaworld_grasp.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld succeeded"
else
    echo "PEBBLE metaworld failed"
    exit 1
fi

. ./tests/run_PEBBLE_metaworld_grasp_inplace.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld grasp_inplace succeeded"
else
    echo "PEBBLE metaworld grasp_inplace failed"
    exit 1
fi

. ./tests/run_PEBBLE_metaworld_reward_beta.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld reward_beta succeeded"
else
    echo "PEBBLE metaworld reward_beta failed"
    exit 1
fi

. ./tests/run_PEBBLE_metaworld_gaussian_beta.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld gaussian succeeded"
else
    echo "PEBBLE metaworld gaussian failed"
    exit 1
fi

. ./tests/run_PEBBLE_metaworld_gaussian_beta_divide.sh
if [ $? -eq 0 ]; then
     echo "PEBBLE metaworld gaussian divide succeeded"
else
    echo "PEBBLE metaworld gaussian failed"
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

. ./tests/run_PrefPPO_dmc.sh
if [ $? -eq 0 ]; then 
    echo "PrefPPO dmc succeeded" 
else
    echo "PrefPPo dmc failed" 
    exit 1
fi

. ./tests/run_PEBBLE_metaworld_box_beta.sh
if [ $? -eq 0 ]; then 
    echo "PEBBLE box beta succeeded" 
else
    echo "PEBBLE box beta failed" 
    exit 1
fi