#!/bin/bash
#SBATCH --job-name=pydylan       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=03:59:59          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=al5844@princeton.edu

module purge
module load anaconda3/2021.11
conda activate pydylan
export LD_LIBRARY_PATH=/home/al5844/.conda/envs/pydylan/lib/python3.8/site-packages/pydylan/bin
python /home/al5844/desktop/project/global_optimization/pydylan-wrapper/simulator/cr3bp_earth_mission_simulator.py  --file_num 20 --sample_num_per_file 100 --process_num 10 --start_bdry=6.3 --end_bdry=8.0 --thrust=1.0 --seed_option 0