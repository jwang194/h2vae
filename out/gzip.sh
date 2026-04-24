#!/bin/bash
#$ -o /u/home/j/jwang194/zp/zaitlen/h2vae/out/logs/gzip.o  
#$ -e /u/home/j/jwang194/zp/zaitlen/h2vae/out/logs/gzip.e
#$ -l h_data=16G
#$ -l time=1:00:00

. /u/local/Modules/default/init/modules.sh
. /u/home/j/jwang194/.profile

cd /u/home/j/jwang194/zp/zaitlen/h2vae/out/${1}/gwas/results/
gzip *
