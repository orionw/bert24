
import wandb
import subprocess
import sys

def train():
   
   # Create readable name from parameters
   run_name = f"mini_lr{wandb.config.lr}_warmup{wandb.config.t_warmup/1e9}B_bsw{wandb.config.batch_size_warmup_tokens/1e9}B"
   
   cmd = [
       "/weka/scratch/bvandur1/oweller2/miniconda3/envs/bert24/bin/composer",
       "main.py",
       "/scratch/bvandur1/oweller2/retrieval_pretraining/bert24/retrieval_pretraining/yamls/pythia-like/mini/decoder_mini.yaml",
       f"train_loader.batch_size_warmup_tokens={wandb.config.batch_size_warmup_tokens}",
       f"optimizer.lr={wandb.config.lr}",
       f"scheduler.t_warmup={wandb.config.t_warmup}",
       f"run_name={run_name}",
       f"save_folder=models_pythia_like/mini_sweep/{run_name}"
   ]
   
   try:
       # Change to working directory
       os.chdir('/home/oweller2/my_scratch/retrieval_pretraining/bert24')
       
       result = subprocess.run(cmd, check=True, capture_output=True, text=True)
       if result.returncode != 0:
           print(f"Command failed with error: {result.stderr}")
           sys.exit(1)
   except subprocess.CalledProcessError as e:
       print(f"Process failed with error: {e}")
       sys.exit(1)

wandb.agent('{sweep_id}', train)
