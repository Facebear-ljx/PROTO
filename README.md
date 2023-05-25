# PROTO: Iterative Policy Regularized Offline-to-Online Reinforcement Learning

## How to run the code

### Install dependencies

These are the same setup instructions as in [Implicit Q-Learning](https://github.com/ikostrikov/implicit_q_learning).

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Example training code

Locomotion
```bash
bash 2online_mujoco.sh
bash 2online_mujoco_td3.sh
```

AntMaze
```bash
bash 2online_antmaze.sh
bash 2online_antmaze_td3.sh
```

Adroit
```bash
bash 2online_adroit.sh
bash 2online_adroit_td3.sh
```