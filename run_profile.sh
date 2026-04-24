#!/bin/bash
#$ -o /u/home/j/jwang194/zp/zaitlen/h2vae/logs/profile.o
#$ -e /u/home/j/jwang194/zp/zaitlen/h2vae/logs/profile.e
#$ -l h_data=128G
#$ -l time=1:00:00
#$ -l gpu
#$ -N profile_hvae

. /u/local/Modules/default/init/modules.sh
. /u/home/j/jwang194/.profile

cd /u/home/j/jwang194/zp/zaitlen/h2vae

~/zp/zaitlen/conda/envs/ccseg/bin/python3 profile_epoch.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/kinship.hdf5 \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --hweights aux/uniform.64.weights \
    --residualize-covariates aux/PC1_40_Age_Sex.covariates \
    --h-weight 1
