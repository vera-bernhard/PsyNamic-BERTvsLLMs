

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

* Install GPU stuff:
```bash
nvidia-smi #to check cuda version
mamba install pytorch pytorch-cuda=<required-version> transformers deepspeed -c pytorch -c nvidia
mamba install -r requirements.txt
python -c 'import tensorflow as tf; print("Built with CUDA:", tf.test.is_built_with_cuda()); print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU"))); print("TF version:", tf.__version__)'
```

* Interactive session
```srun --pty -n 1 -c 2 --time=00:15:00 --gpus=A100:1 bash -l```

* Account
``` sacctmgr show assoc format=account%30,partition,user,qos%30 user=$USER
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