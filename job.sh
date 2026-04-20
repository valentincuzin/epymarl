#!/bin/bash

#SBATCH --job-name=dicg-de-mappo-hp    # Nom du job
#SBATCH --output=dicg-de-mappo-hp_%j.log   # Standard output et error log

#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --cpus-per-task=1             # Utiliser un seul CPU pour cette tâche (job)
#SBATCH --mem=32G                  # Size of cpu memory
#SBATCH --time=0-20:00:00         # Max duration days-hours:minutes:seconds

#SBATCH --mail-user=valentin.cuzin-rambaud@etu.univ-lyon1.fr  # Where to send mail
#SBATCH --mail-type=FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

# next line is important to stop the submission script at the 1st error
set -e

# activate recent Python
module load Programming_Languages/micromamba/2.0.5
source ~/.bashrc
micromamba activate p313

cd ~/epymarl/
# run Python script
python src/main.py --hp_search=25 --seed=0 --config=dicg-de-mappo --env-config=gymma with env_args.key="mpe2-simple-tag-v3" env_args.pretrained_wrapper="RandomTag"