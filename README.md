# Deep off-policy ILC

This repository contains the implementation for the paper [Deep off-policy Iterative Learning Control](https://openreview.net/forum?id=Bi0E3lbvnU) (L4DC 2023).

The paper proposes an update equation for the value-function gradients to speed up actor critic methods in reinforcement learning. This update is inspired by iterative learning control (ILC) approaches that use approximate simulator gradients to speed up optimization. The value-gradient update leads to a significant improvement in sample efficiency and sometimes better performance both when learning from scratch or fine-tuning a pre-trained policy in a new environment. We add this update to soft actor-critic (SAC) and demonstrate improvements.

#### Prerequisites

- In the project folder, create a virtual environment in Anaconda:

  ```
  conda env create -f deepilc.yml
  conda activate deepilc
  ```

- dflex

  ```
  cd dflex
  pip install -e .
  ```


## Training

Running the following command to train rexquadrotor with VG-SAC.

```
python main.py --env-name rexquadrotor --jacobian --cuda 
```

Use the following flags to train the model with different settings:
```
--env-name : {cartpole, acrobot, rexquadrotor, halfcheetah, hopper, DmcCheetah, DmcHopper}
--jacobian : Train with approximate model jacobians
--pretrain : Pretrain the model in the approximate model(default: False)
--pretrain_jacobian : Pretrain the model in the approximate model with model jacobians (default: False)
--zeroth : Train with zeroth order jacobians from the approximate model (default: False)
```

## Citation

If you find our paper or code is useful, please consider citing:
```
  @inproceedings{
	gurumurthy2023valuegradien,
	title={Deep Off-policy Iterative Learning Control},
	author={Swaminathan Gurumurthy and J Zico Kolter and Zachary Manchester},
	booktitle={5th Annual Learning for Dynamics {\&} Control Conference},
	year={2023},
	url={https://openreview.net/forum?id=Bi0E3lbvnU}
  }
```

## Acknowledgements

The code for the SAC implementation in the repo has been adapted from https://github.com/pranz24/pytorch-soft-actor-critic . 
The code for various environments used in the repo have been adapted from https://github.com/NVlabs/DiffRL, https://github.com/bjack205/BilinearControl.jl and https://github.com/sgillen/ssac