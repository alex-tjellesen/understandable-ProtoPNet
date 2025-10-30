#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J responsible_ai_no_pp
### Num cores
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
### Memory per core
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ~/understandable-ProtoPNet/job_outs/gpu-%J.out
#BSUB -e ~/understandable-ProtoPNet/job_outs/gpu_%J.err
# -- end of LSF options --

module swap python3/3.11.9
module swap cuda/12.6.3
 
cd ~/understandable-ProtoPNet
source .venv/bin/activate

python train.py --dataset /zhome/ea/6/187439/understandable-ProtoPNet/datasets/cub200 \
 --exp_name no_ppnet \
 --epochs 501 \
 --batch_size 128 \
 --num_workers 8 \
 --seed 42 \
 --warm_epochs 50 \
 --test_interval 50 \
 --no_ppnet

git add .
git commit -m "Completed training run: no_ppnet"
git push origin hpc_run