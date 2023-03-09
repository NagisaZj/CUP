# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gym
import mtenv
from gym.vector.async_vector_env import AsyncVectorEnv

from mtrl.env.vec_env import MetaWorldVecEnv, VecEnv
from mtrl.utils.types import ConfigType


def build_dmcontrol_vec_env(
    domain_name: str,
    task_name: str,
    prefix: str,
    make_kwargs: ConfigType,
    env_id_list: List[int],
    seed_list: List[int],
    mode_list: List[str],
) -> VecEnv:
    def get_func_to_make_envs(seed: int, initial_task_state: int):
        def _func() -> mtenv.MTEnv:
            kwargs = deepcopy(make_kwargs)
            kwargs["seed"] += seed
            kwargs["initial_task_state"] = initial_task_state
            return mtenv.make(
                f"MT-HiPBMDP-{domain_name.capitalize()}-{task_name.capitalize()}-vary-{prefix.replace('_', '-')}-v0",
                **kwargs,
            )

        return _func

    funcs_to_make_envs = [
        get_func_to_make_envs(seed=seed, initial_task_state=task_state)
        for (seed, task_state) in zip(seed_list, env_id_list)
    ]

    env_metadata = {"ids": env_id_list, "mode": mode_list}

    env = VecEnv(env_metadata=env_metadata, env_fns=funcs_to_make_envs, context="spawn")

    return env


def build_metaworld_vec_env(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    from mtenv.envs.metaworld.env import (
        get_list_of_func_to_make_envs as get_list_of_func_to_make_metaworld_envs,
        get_list_of_func_to_make_envs_ML as get_list_of_func_to_make_metaworld_envs_ML,
    )
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    # num_tasks = int(benchmark_name.replace("MT", ""))
    num_tasks = -1
    # nc = 10 if config.env.num_envs ==30 else 1
    # nc = 10 if (num_tasks == 10 and config.env.num_envs !=30) else nc
    nc = 10 if  (config.env.num_envs ==10 or config.env.num_envs==30) else 1
    nc = 30 if (config.env.num_envs == 30) else nc
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name,
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": nc,
        "should_perform_reward_normalization": False,
        "fix_goal":config.env.fix_goal,
        "task_name":config.env.tasks[0],
        "task_idx":None if config.env.task_idx==-1 else config.env.task_idx
    }
    if 'MT' in benchmark_name:
        funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
            **make_kwargs
        )
        env_dict_tmp = {}
        # if config.env.num_envs ==3 or config.env.num_envs ==30:
        #     num_tasks = 3
        #     funcs_to_make_envs = funcs_to_make_envs[:3*10]
        #     for k, i in env_id_to_task_map.items():
        #         if k in ['reach-v2', 'push-v2', 'pick-place-v2']:
        #             env_dict_tmp[k] = i
        #     env_id_to_task_map = env_dict_tmp
        if "MT" in benchmark_name:
            if hasattr(config.env,'tasks'):
                num_tasks = len(config.env.tasks)
                funcs_to_make_envs_tmp = []
                for i in config.env.ids:
                    funcs_to_make_envs_tmp.extend(funcs_to_make_envs[i*nc:(i+1)*nc])
                funcs_to_make_envs = funcs_to_make_envs_tmp
            env_dict_tmp = {}
            if hasattr(config.env,'tasks'):
                num_tasks = len(config.env.tasks)
                env_dict_tmp = {}
                for k, i in env_id_to_task_map.items():
                    if k in config.env.tasks:
                        env_dict_tmp[k] = i
                env_id_to_task_map = env_dict_tmp
        elif "ML1" in benchmark_name:
            if hasattr(config.env,'tasks'):
                num_tasks = len(config.env.tasks)
    else:
        funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
            **make_kwargs
        )
        num_tasks = len(config.env.tasks)
        # num_tasks = len(config.env.task_idx)

    print(env_id_to_task_map)
    cnt = 0
    for k, i in env_id_to_task_map.items():
        print(cnt,k)
        cnt+=1
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    return env, env_id_to_task_map


def make_env_mine(task_name):
    def get_func_to_make_envs(task_name):
        def _make_env():
            env = gym.make()
            return env

        return _make_env

    return None,None


# def make_extra_envs(
#     config
# ):
#
#     nc = 10
#     make_kwargs = {
#         "task_name":config.env.tasks[0]
#     }
#     if 1:
#         funcs_to_make_envs, env_id_to_task_map = make_env_mine(
#             **make_kwargs
#         )
#         env_dict_tmp = {}
#         # if config.env.num_envs ==3 or config.env.num_envs ==30:
#         #     num_tasks = 3
#         #     funcs_to_make_envs = funcs_to_make_envs[:3*10]
#         #     for k, i in env_id_to_task_map.items():
#         #         if k in ['reach-v2', 'push-v2', 'pick-place-v2']:
#         #             env_dict_tmp[k] = i
#         #     env_id_to_task_map = env_dict_tmp
#         if "MT" in benchmark_name:
#             if hasattr(config.env,'tasks'):
#                 num_tasks = len(config.env.tasks)
#                 funcs_to_make_envs_tmp = []
#                 for i in config.env.ids:
#                     funcs_to_make_envs_tmp.extend(funcs_to_make_envs[i*nc:(i+1)*nc])
#                 funcs_to_make_envs = funcs_to_make_envs_tmp
#             env_dict_tmp = {}
#             if hasattr(config.env,'tasks'):
#                 num_tasks = len(config.env.tasks)
#                 env_dict_tmp = {}
#                 for k, i in env_id_to_task_map.items():
#                     if k in config.env.tasks:
#                         env_dict_tmp[k] = i
#                 env_id_to_task_map = env_dict_tmp
#         elif "ML1" in benchmark_name:
#             if hasattr(config.env,'tasks'):
#                 num_tasks = len(config.env.tasks)
#     else:
#         funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
#             **make_kwargs
#         )
#         num_tasks = len(config.env.tasks)
#         # num_tasks = len(config.env.task_idx)
#
#     print(env_id_to_task_map)
#     cnt = 0
#     for k, i in env_id_to_task_map.items():
#         print(cnt,k)
#         cnt+=1
#     env_metadata = {
#         "ids": list(range(num_tasks)),
#         "mode": [mode for _ in range(num_tasks)],
#     }
#     env = MetaWorldVecEnv(
#         env_metadata=env_metadata,
#         env_fns=funcs_to_make_envs,
#         context="spawn",
#         shared_memory=False,
#     )
#     return env, env_id_to_task_map