#!/usr/bin/env python

timestamp_files = ['timestamps_A.csv', 'timestamps_B.csv', 'timestamps_C.csv']
obs_array = range(5, 65, 5)

for timestamp_file in timestamp_files:
    for nobs in obs_array:
        print("./simulate_observations.py {} setups/single.py {}".format(timestamp_file, nobs))
