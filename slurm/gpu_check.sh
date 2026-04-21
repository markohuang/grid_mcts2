#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/gpu_check_%j.out

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

echo "host=$(hostname)"
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
