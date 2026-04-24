#!/bin/bash
#$ -o /u/home/j/jwang194/zp/zaitlen/h2vae/logs/train_hvae.o
#$ -e /u/home/j/jwang194/zp/zaitlen/h2vae/logs/train_hvae.e
#$ -l h_rt=24:00:00
#$ -l gpu

. /u/local/Modules/default/init/modules.sh
. /u/home/j/jwang194/.profile

cd /u/home/j/jwang194/zp/zaitlen/h2vae

mambaload ccseg

python3 train_hvae.py "$@"
