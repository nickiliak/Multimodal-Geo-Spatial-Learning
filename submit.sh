#!/bin/sh
## General options
## -- specify queue --
#BSUB -q gpuv100
## -- set the job Name --
#BSUB -J advmachlearn
## -- ask for number of cores (default: 1) --
#BSUB -n 4
## -- specify that the cores MUST BE on a single host --
#BSUB -R "span[hosts=1]"
## -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
## -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:30
## request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o advmach_%J.out
#BSUB -e advmach_%J.err
# -- end of LSF options --
#here load the modules, and activate the environment if needed

module load python3
source .venv/bin/activate
torchrun --standalone --nproc_per_node=1 notebooks/team/training.py