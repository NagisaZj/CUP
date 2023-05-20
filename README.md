[![CircleCI](https://circleci.com/gh/facebookresearch/mtrl.svg?style=svg&circle-token=8cc8eb1b9666a65e27a21c39b5d5398744365894)](https://circleci.com/gh/facebookresearch/mtrl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# CUP: Critic-Guided Policy Reuse
This repository is the official implementation of CUP: Critic-Guided Policy Reuse], which has been accepted by NeurIPS 2022. Please create an issue if you have any problems!

## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Usage](#Usage)


## Introduction




## Setup

* Install dependencies: 
  ```
  git init
  git add .
  git commit -m init
  pip install -r requirements/dev.txt
  cd ./src
  git clone git@github.com:NagisaZj/metaworld-cup.git
  git clone git@github.com:NagisaZj/mtenv.git
  cd ./src/mtenv
  pip install -e .
  cd ../metaworld-cup
  pip install -e .
  ```
## Usage
CUP:

<font color='red'> CAUTION: Remember to replace setup.load_dir,  setup.load_dir_2, and setup.load_dir_3 with your own absolute path to the corresponding directories.</font>

  ```
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-push-back \
env.task_idx=-1 \
env.fix_goal=0 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=1000000 \
setup.seed=1695 \
experiment.eval_freq=5000 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=1 \
agent.multitask.should_use_disentangled_alpha=True \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.should_use_disjoint_policy=False \
agent.multitask.should_use_task_encoder=True \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True \
setup.relabel_num_tasks=1 \
setup.relabel_range=10 \
setup.load_dir=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model \
setup.load_dir_2=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model \
setup.load_dir_3=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_253/model \
setup.load=1 \
setup.load_log_std_bounds=[-20,2]
  ```

SAC baseline: Just set agent.use_expert to 0 in the corresponding config file (config/agent/state_sac.yaml), or pass arguments with commands, for example:
  ```
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-push-back \
env.task_idx=-1 \
env.fix_goal=0 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=1000000 \
setup.seed=1695 \
experiment.eval_freq=5000 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=1 \
agent.multitask.should_use_disentangled_alpha=True \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.should_use_disjoint_policy=False \
agent.multitask.should_use_task_encoder=True \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True \
setup.relabel_num_tasks=1 \
setup.relabel_range=10 \
setup.load_dir=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model \
setup.load_dir_2=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_43_2/model \
setup.load_dir_3=/data3/zj/CUP/source_policies/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_253/model \
setup.load=1 \
setup.load_log_std_bounds=[-20,2] \
agent.multitask.use_expert=0
  ```

Other available environments can be seen in ./config/env.