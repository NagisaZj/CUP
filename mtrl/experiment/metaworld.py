# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import Dict

import hydra
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import multitask
from mtrl.utils.types import ConfigType


class Experiment(multitask.Experiment):
    """Experiment Class"""

    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        super().__init__(config, experiment_id)
        self.should_reset_env_manually = True

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if self.config.env.benchmark._target_ in [
                "metaworld.ML1",
                "metaworld.MT1",
                "metaworld.MT10",
                "metaworld.MT50",
                "metaworld.ML10",
                "metaworld.ML50",
            ]:
                eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))
            else:
                raise ValueError(
                    f"`{self.config.env.benchmark._target_}` env is not supported by metaworld experiment."
                )
        return eval_modes_to_env_ids

    def build_envs(self):
        # benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        if ('v2' not in self.config.env.tasks[0] )or ('LunarLander' in self.config.env.tasks[0]):
            envs,metadata = env_builder.make_extra_envs(self.config)
            return envs, metadata
        else:
            benchmark = 0
            envs = {}
            mode = "train"
            envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
                config=self.config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
            )
            print(env_id_to_task_map.keys())
            mode = "eval"
            envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
                config=self.config,
                benchmark=benchmark,
                mode="train",
                env_id_to_task_map=env_id_to_task_map,
            )
            # In MT10 and MT50, the tasks are always sampled in the train mode.
            # For more details, refer https://github.com/rlworkgroup/metaworld

            max_episode_steps = 500
            # hardcoding the steps as different environments return different
            # values for max_path_length. MetaWorld uses 150 as the max length.
            metadata = self.get_env_metadata(
                env=envs["train"],
                max_episode_steps=max_episode_steps,
                ordered_task_list=list(env_id_to_task_map.keys()),
            )
            return envs, metadata

    def create_env_id_to_index_map(self) -> Dict[str, int]:
        env_id_to_index_map: Dict[str, int] = {}
        current_id = 0
        for env in self.envs.values():
            assert isinstance(env, VecEnv)
            for env_name in env.ids:
                if env_name not in env_id_to_index_map:
                    env_id_to_index_map[env_name] = current_id
                    current_id += 1
        return env_id_to_index_map

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        episode_step = 0
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        agent = self.agent
        offset = self.config.experiment.num_eval_episodes

        while episode_step < self.max_episode_steps:
            # print(multitask_obs['task_obs'])
            for i in range(self.config.setup.relabel_num_tasks):
                for j in range(self.config.setup.relabel_range):
                    multitask_obs['task_obs'][i * self.config.setup.relabel_range + j] = i
            # print(multitask_obs['task_obs'])
            # if episode_step<300:
            #     for j in range(10):
            #         multitask_obs['task_obs'][20+j] = 1
            # else:
            #     for j in range(10):
            #         multitask_obs['task_obs'][20+j] = 0
            with agent_utils.eval_mode(agent):
                action = agent.select_action(
                    multitask_obs=multitask_obs, modes=["eval"]
                )
            multitask_obs, reward, done, info = vec_env.step(action)
            success += np.asarray([x["success"] for x in info])
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        start_index = 0
        # print(success)
        success = (success > 0).astype("float")
        for mode in self.eval_modes_to_env_ids:
            num_envs = len(self.eval_modes_to_env_ids[mode])
            self.logger.log(
                f"{mode}/episode_reward",
                episode_reward[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            self.logger.log(
                f"{mode}/success",
                success[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            for _current_env_index, _current_env_id in enumerate(
                self.eval_modes_to_env_ids[mode]
            ):
                # print(self.eval_modes_to_env_ids['eval'])
                self.logger.log(
                    f"{mode}/episode_reward_env_index_{_current_env_index}",
                    episode_reward[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].mean(),
                    step,
                )
                self.logger.log(
                    f"{mode}/success_env_index_{_current_env_index}",
                    success[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].mean(),
                    step,
                )
                self.logger.log(
                    f"{mode}/env_index_{_current_env_index}", _current_env_id, step
                )
            start_index += offset * num_envs
        self.logger.dump(step)
        return success.mean()
        # self.logger.dump(step)

    def plot_evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        episode_step = 0
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        agent = self.agent
        offset = self.config.experiment.num_eval_episodes
        grasp = [0]
        Q_s = np.zeros([500,4])
        while episode_step < self.max_episode_steps:
            # print(multitask_obs['task_obs'])
            for i in range(self.config.setup.relabel_num_tasks):
                for j in range(self.config.setup.relabel_range):
                    multitask_obs['task_obs'][i * self.config.setup.relabel_range + j] = 0
            # print(multitask_obs['task_obs'])
            # if episode_step<300:
            #     for j in range(10):
            #         multitask_obs['task_obs'][20+j] = 1
            # else:
            #     for j in range(10):
            #         multitask_obs['task_obs'][20+j] = 0
            with agent_utils.eval_mode(agent):
                action = agent.select_action(
                    multitask_obs=multitask_obs, modes=["eval"]
                )
            weights=agent.get_weights(multitask_obs)
            for y in range(4):
                Q_s[episode_step,y] = weights[y]
            multitask_obs, reward, done, info = vec_env.step(action)
            grasp.append(50 if info[0]['grasp_success'] else 0)
            success += np.asarray([x["success"] for x in info])
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        start_index = 0
        import matplotlib.pyplot as plt
        plt.figure()
        # plt.plot(np.arange(500),grasp[:500])

        def smooth(data, smooth_range):
            # print('hhhhhhh', type(data), len(data))
            new_data = np.zeros_like(data)
            for i in range(0, data.shape[-1]):
                if i < smooth_range:
                    new_data[:, i] = 1. * np.sum(data[:, :i + 1], axis=1) / (i + 1)
                else:
                    new_data[:, i] = 1. * np.sum(data[:, i - smooth_range + 1:i + 1], axis=1) / smooth_range

            return new_data

        Q_s = smooth(Q_s.transpose(),5).transpose()
        labels = ['Target Policy','Reach','Push','Pick-Place']
        color_set = {
            'Amaranth': np.array([0.9, 0.17, 0.31]),  # main algo
            'Amber': np.array([1.0, 0.49, 0.0]),  # main baseline
            'Bleu de France': np.array([0.19, 0.55, 0.91]),
            'Electric violet': np.array([0.56, 0.0, 1.0]),
            'Dark sea green': 'forestgreen',
            'Dark electric blue': 'deeppink',
            'Dark gray': np.array([0.66, 0.66, 0.66]),
            'Arsenic': np.array([0.23, 0.27, 0.29]),
        }

        color_list = []
        for key, value in color_set.items():
            color_list.append(value)
        for j in range(4):
            plt.plot(np.arange(500), Q_s[:,j]-Q_s[:,0],label=labels[j], linewidth=3,c=color_list[j])
        plt.legend(loc='best', prop={'size': 26.0}, frameon=True, ncol=1)
        plt.tick_params(labelsize=17)
        plt.title('Pick-Place-Wall', fontdict={'fontsize': 30})
        plt.xlabel('Environment Steps', fontsize=20)
        plt.show()
        # print(success)
        success = (success > 0).astype("float")
        for mode in self.eval_modes_to_env_ids:
            num_envs = len(self.eval_modes_to_env_ids[mode])
            self.logger.log(
                f"{mode}/episode_reward",
                episode_reward[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            self.logger.log(
                f"{mode}/success",
                success[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            for _current_env_index, _current_env_id in enumerate(
                self.eval_modes_to_env_ids[mode]
            ):
                # print(self.eval_modes_to_env_ids['eval'])
                self.logger.log(
                    f"{mode}/episode_reward_env_index_{_current_env_index}",
                    episode_reward[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].mean(),
                    step,
                )
                self.logger.log(
                    f"{mode}/success_env_index_{_current_env_index}",
                    success[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].mean(),
                    step,
                )
                self.logger.log(
                    f"{mode}/env_index_{_current_env_index}", _current_env_id, step
                )
            start_index += offset * num_envs
        self.logger.dump(step)
        return success.mean()
        # self.logger.dump(step)

    def hierarhichal_evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        for m in range(1,3):
            episode_step = 0
            # for mode in self.eval_modes_to_env_ids:
            #     self.logger.log(f"{mode}/episode", episode, step)

            episode_reward, mask, done, success = [
                np.full(shape=vec_env.num_envs, fill_value=fill_value)
                for fill_value in [0.0, 1.0, False, 0.0]
            ]  # (num_envs, 1)
            multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
            agent = self.agent
            offset = self.config.experiment.num_eval_episodes

            while episode_step < self.max_episode_steps:
                # print('!!!',multitask_obs['task_obs'])
                for i in range(3):
                    for j in range(10):
                        multitask_obs['task_obs'][i * 10 + j] = (i+m)%3
                # print('!!!!',multitask_obs['task_obs'])
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        multitask_obs=multitask_obs, modes=["eval"]
                    )
                multitask_obs, reward, done, info = vec_env.step(action)
                success += np.asarray([x["success"] for x in info])
                mask = mask * (1 - done.astype(int))
                episode_reward += reward * mask
                episode_step += 1
            start_index = 0
            success = (success > 0).astype("float")
            for mode in self.eval_modes_to_env_ids:
                num_envs = len(self.eval_modes_to_env_ids[mode])
                for _current_env_index in range(0,30):
                    self.logger.log(
                        f"{mode}/episode_reward_env_index_{m}{_current_env_index}",
                        episode_reward[
                            start_index
                            + _current_env_index * offset : start_index
                            + (_current_env_index + 1) * offset
                        ].mean(),
                        step,
                    )
                    self.logger.log(
                        f"{mode}/success_env_index_{m}{_current_env_index}",
                        success[
                            start_index
                            + _current_env_index * offset : start_index
                            + (_current_env_index + 1) * offset
                        ].mean(),
                        step,
                    )
                start_index += offset * num_envs
        self.logger.dump(step)

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        for _ in range(num_steps):
            with agent_utils.eval_mode(self.agent):
                action = self.agent.sample_action(
                    multitask_obs=multitask_obs, mode="train"
                )  # (num_envs, action_dim)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
            episode_reward += reward

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                self.replay_buffer.add(
                    multitask_obs["env_obs"][index],
                    action[index],
                    reward[index],
                    next_multitask_obs["env_obs"][index],
                    done_bool,
                    env_index=env_index,
                )

            multitask_obs = next_multitask_obs
            episode_step += 1
