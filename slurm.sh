#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=6-22:10:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is  :$SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR
###################conda environment path ####################################################
source /nlsasfs/home/ttbhashini/prathosh/anaconda3/bin/activate

echo "conda info --envs"

conda activate int-mod

python train.py --t_lang Hindi --s_lang Malayalam
python train.py --t_lang Hindi --s_lang Marathi
python train.py --t_lang Hindi --s_lang Odia
python train.py --s_lang Punjabi --t_lang Hindi
