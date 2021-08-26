#!/bin/bash
#SBATCH --job-name=gpu_multi          # nom du job
#SBATCH --partition=gpu_p2         # de-commente pour la partition gpu_p2
##SBATCH -C v100-32g
##SBATCH --exclusive
##SBATCH --gres=gpu:8 a tester +  avec gpu=2   ntasks-per-node=2 
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1         # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1               # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=3          # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --qos=qos_gpu-t3             # SOQ
          
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=testmultigpu.out      # nom du fichier de sortie
#SBATCH --error=testmultigpuerror.out       # nom du fichier d'erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
module load anaconda-py3/2020.11
conda activate dp

# chargement des modules

 
# echo des commandes lancees
set -x
 
# execution du code

#PL_TORCH_DISTRIBUTED_BACKEND=nccl python train.py trainer.gpus=2 trainer.accelerator=ddp

python train.py trainer.gpus=1 data.train_csv='./' data.human_train_csv='./'  data.val_csv='./' data.human_val_csv='./' data.train_dir='./'
