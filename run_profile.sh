#!/bin/bash
#$ -o /u/home/j/jwang194/zp/zaitlen/h2vae/logs/profile.o
#$ -e /u/home/j/jwang194/zp/zaitlen/h2vae/logs/profile.e
#$ -l h_data=128G
#$ -l time=2:00:00
#$ -l gpu
#$ -N profile_hvae

. /u/local/Modules/default/init/modules.sh
. /u/home/j/jwang194/.profile

cd /u/home/j/jwang194/zp/zaitlen/h2vae

# Profile the NIfTI / vae3d branch on the production T1 MRI cohort.
# Two epochs are run; the first is discarded as warmup.
~/zp/zaitlen/conda/envs/ccseg/bin/python3 profile_epoch.py \
    --model vae3d \
    --images data/images/t1.tsv \
    --genetics data/genetics/mri_kinship.hdf5 \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --hweights aux/uniform.128.weights \
    --decode-covariates aux/PC1_40_Age_Sex.covariates \
    --residualize-covariates aux/ICV.covariates \
    --kinship \
    --zdim 128 \
    --bs 8 \
    --h-weight 1 \
    --profile-epochs 2
