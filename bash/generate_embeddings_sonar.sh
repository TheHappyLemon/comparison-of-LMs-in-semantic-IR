#!/bin/bash

#PBS -N gen_embd_sonar
#PBS -q batch
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe
#PBS -W x=HOSTLIST:wn68

REPO_URL="https://github.com/TheHappyLemon/comparison-of-LLMs-in-semantic-IR.git"
CODE_DIR="./Code"
PYTHON_SCRIPT="generate_embeddings_sonar.py"

conda deactivate
module purge

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

echo "Cloning repository!"
module load git
git clone $REPO_URL
REPO_NAME=$(basename $REPO_URL .git)
echo "Cloned repo!"

echo Changing directory to $REPO_NAME
cd $REPO_NAME
git pull
cd $CODE_DIR
module load anaconda/conda-23.1.0
source activate sonar_env

python3 generate_embeddings_sonar.py
echo "Current conda env info:" 
echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX
echo "try to check sonar version:"
python3 -c "import sonar; print(sonar.__version__)"

conda deactivate
module purge
echo "Script execution completed."

