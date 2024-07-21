sbatch  --mail-type=begin --mail-type=end --mail-type=fail --mail-user=edfan@email.unc.edu -N 1 -n 1 -p a100-gpu --mem=128g -t 02-00:00:00 --qos gpu_access --gres=gpu:nvidia_a100-pcie-40gb:3 --wrap="python -m experiments.run_want"

sbatch  --mail-type=begin --mail-type=end --mail-type=fail --mail-user=edfan@email.unc.edu -N 1 -n 1 -p a100-gpu --mem=128g -t 02-00:00:00 --qos gpu_access --gres=gpu:nvidia_a100-pcie-40gb:2 --wrap="python -m experiments.run_memorization_llama"

sbatch  --mail-type=begin --mail-type=end --mail-type=fail --mail-user=edfan@email.unc.edu -N 1 -n 1 -p a100-gpu --mem=128g -t 02-00:00:00 --qos gpu_access --gres=gpu:nvidia_a100-pcie-40gb:3 --wrap="python -m experiments.run_want"

sbatch  --mail-type=begin --mail-type=end --mail-type=fail --mail-user=edfan@email.unc.edu -N 1 -n 1 -p l40-gpu --mem=128g -t 02-00:00:00 --qos gpu_access --gres=gpu:nvidia_l40s:5 --wrap="python -m experiments.run_want"