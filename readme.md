

# Science Cluster Stuff

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

  * What has previously worked 
    ```bash
    nvidia-smi #to check cuda version
    mamba install pytorch pytorch-cuda=<required-version> transformers deepspeed -c pytorch -c nvidia
    mamba install -r requirements.txt
    python -c 'import tensorflow as tf; print("Built with CUDA:", tf.test.is_built_with_cuda()); print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU"))); print("TF version:", tf.__version__)'
    ```

  * New idea: 
      Checkout versions here: https://pytorch.org/get-started/previous-versions/

      for CUDA 12.6
      ```
      pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu12
      ```

# Uebelix
  ```
  sbatch code-tunnel.sbatch
  salloc --partition=gpu-invest --qos=job_gpu_preemptable --gres=gpu:a100:1 --mem-per-gpu=80G --time=01:30:00
  srun --pty -n 1 -c 2 --partition=gpu-invest --qos=job_gpu_preemptable --gres=gpu:a100:1 --mem-per-gpu=80G --time=01:30:00 bash -l

  module load Anaconda3
  nvidia-smi
  module load CUDA/12.3.0
  conda install pytorch::pytorch torchvision torchaudio -c pytorch
  python -c 'import torch as t; print("is available: ", t.cuda.is_available()); print("device count: ", t.cuda.device_count()); print("current device: ", t.cuda.current_device()); print("cuda device: ", t.cuda.device(0)); print("cuda device name: ", t.cuda.get_device_name(0)); print("cuda version: ", t.version.cuda)'
  ```