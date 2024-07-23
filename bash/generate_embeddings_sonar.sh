#!/bin/bash

#PBS -N gen_embd_sonar
#PBS -q batch
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe

REPO_URL="https://github.com/TheHappyLemon/comparison-of-LLMs-in-semantic-IR.git"
CODE_DIR="./Code"
VENV_NAME="sonar_env"
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

module load git
git clone $REPO_URL
REPO_NAME=$(basename $REPO_URL .git)

# Check if the repo was cloned successfully
if [ ! -d "$REPO_NAME" ]; then
    echo "Failed to clone the repository."
    exit 1
fi
echo "Cloned repo!"

cd $REPO_NAME
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
echo "Created virtual env!"

pip install fairseq2
pip install sonar-space
pip install h5py
echo "Installed modules!"

cd $CODE_DIR
python3 $PYTHON_SCRIPT

deactivate
echo "Script execution completed."
