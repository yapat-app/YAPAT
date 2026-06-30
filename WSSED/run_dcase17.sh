#!/bin/bash
pool_list="max_pool avg_pool attention_pool linear_pool exp_pool auto_pool power_pool hi_pool"
for pool in $pool_list:
do
	python run_dcase17.py -w 16 -b 256 -e 100 -lr 0.0003 -g 0 -p $pool
done
