
# UZH Science Cluster Stuff

* Make virtual environment
```bash
module load mamba
mamba activate ma-env
mamba env list
```

* Module Load, unload, list
```bash
module load a100
module unload a100
module list 
```

* Interactive session
```bash
srun --pty -n 1 -c 2 --time=00:30:00 --mem=16G --gpus=A100:1 bash -l
```

* Account
```bash
sacctmgr show assoc format=account%30,partition,user,qos%30 user=$USER
sacctmgr show user $USER format=defaultaccount%30
```

* Resources:
Available GPUs: H100, A100, V100, T4.

- **H100**: Latest generation, top performance for large-scale AI training.
- **A100**: High performance, versatile for training and inference.
- **V100**: Solid for general-purpose deep learning workloads.
- **T4**: Optimized for inference and light training tasks.


# Unibe Uebelix stuff
* Connect to Server and make tunnel for VSCode
```
ssh -i ~/.ssh/id_ed25519.pub vb25l522@submit01.unibe.ch
sbatch code-tunnel.sbatch
```

* Uplaod Llama model
```
rsync -avz me-llama/ vb25l522@submit01.unibe.ch:/storage/homefs/vb25l522/me-llama/
```

* Befor running script 
```
module load Anaconda3
module load CUDA/12.3.0
conda activate ma_env
```

# Other notes

* Run python script as modules
```
python -m zero_shot.predict_zero_shot
```

* Pytorch / Cuda Virtual Environment Issues

   * Instructions from Science Cluster Documentation:
    ```
    mamba create -n torch -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda
    ```

  * What has worked this time round 
    To check required CUDA version (shows highest supported CUDA version from GPU)
    ```
    nvidia-smi
    ```

    Checkout versions here: https://pytorch.org/get-started/previous-versions/

    e.g. for CUDA 12.6
    ```
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu12
    ```