#!/bin/bash

#PBS -N gen_embd_sonar
#PBS -q batch
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe

REPO_URL="https://github.com/TheHappyLemon/comparison-of-LLMs-in-semantic-IR.git"
CODE_DIR="./Code"
PYTHON_SCRIPT="generate_embeddings_sonar.py"

# -- Switching to the directory from which the "qsub" command was run, moving to actual working directory
# -- Make sure any symbolic links are resolved to absolute path 
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)  ##ag
cd $PBS_O_WORKDIR
echo Working directory is: $PBS_O_WORKDIR
# --Calculate the number of processors and nodes allocated to this run.
NPROCS=`wc -l < $PBS_NODEFILE`
NNODES=`uniq $PBS_NODEFILE | wc -l`

# -- Displaying job information
echo Running on host `hostname`
echo Time is `date`
echo Current working directory is `pwd`
echo "Node file: $PBS_NODEFILE :"
echo "---------------------"
cat $PBS_NODEFILE
echo "---------------------"
echo Using ${NPROCS} processors across ${NNODES} nodes

if [ ! -d "$REPO_NAME" ]; then
    echo "Cloning repository!"
    module load git
    git clone $REPO_URL
    REPO_NAME=$(basename $REPO_URL .git)
    echo "Cloned repo!"
fi

echo Changing directory to $REPO_NAME
cd $REPO_NAME
module load anaconda/conda-23.1.0
conda activate conda_env_sonar

cd $CODE_DIR
python3 $PYTHON_SCRIPT

conda deactivate
echo "Script execution completed."

# This all in conda env already!
# pip install fairseq2
# pip install sonar-space
# pip install h5py
# and also I need to have 'libsndfile1', because:
# File "/mnt/beegfs2/home/artjom01/wrk/comparison-of-LLMs-in-semantic-IR/sonar_env/lib/python3.10/site-packages/fairseq2n/__init__.py", line 58, in _load_sndfile
    #raise OSError(
#OSError: libsndfile is not found! Since you are in a Conda environment, use `conda install -c conda-forge libsndfile==1.0.31` to install it.

