# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a multi-task model."""

import time
from typing import Dict, List, Tuple

import hydra
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv  # type: ignore
from mtrl.experiment import experiment
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType
import torch, copy


class Experiment(experiment.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a multi-task model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)
        self.eval_modes_to_env_ids = self.create_eval_modes_to_env_ids()
        self.should_reset_env_manually = False
        self.metrics_to_track = {
            x[0] for x in self.config.metrics["train"] if not x[0].endswith("_")
        }

    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Build environments and return env-related metadata"""
        if "dmcontrol" not in self.config.env.name:
            raise NotImplementedError
        envs: EnvsDictType = {}
        mode = "train"
        env_id_list = self.config.env[mode]
        num_envs = len(env_id_list)
        seed_list = list(range(1, num_envs + 1))
        mode_list = [mode for _ in range(num_envs)]

        envs[mode] = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )
        envs["eval"] = self._create_dmcontrol_vec_envs_for_eval()
        metadata = self.get_env_metadata(env=envs["train"])
        return envs, metadata

    def create_eval_modes_to_env_ids(self) -> Dict[str, List[int]]:
        """Map each eval mode to a list of environment index.

            The eval modes are of the form `eval_xyz` where `xyz` specifies
            the specific type of evaluation. For example. `eval_interpolation`
            means that we are using interpolation environments for evaluation.
            The eval moe can also be set to just `eval`.

        Returns:
            Dict[str, List[int]]: dictionary with different eval modes as
                keys and list of environment index as values.
        """
        eval_modes_to_env_ids: Dict[str, List[int]] = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            # todo: add support for mode == "eval"
            if "_" in mode:
                _mode, _submode = mode.split("_")
                env_ids = self.config.env[_mode][_submode]
                eval_modes_to_env_ids[mode] = env_ids
            elif mode != "eval":
                raise ValueError(f"eval mode = `{mode}`` is not supported.")
        return eval_modes_to_env_ids

    def _create_dmcontrol_vec_envs_for_eval(self) -> EnvType:
        """Method to create the vec env with multiple copies of the same
        environment. It is useful when evaluating the agent multiple times
        in the same env.

        The vec env is organized as follows - number of modes x number of tasks per mode x number of episodes per task

        """

        env_id_list: List[str] = []
        seed_list: List[int] = []
        mode_list: List[str] = []
        num_episodes_per_env = self.config.experiment.num_eval_episodes
        for mode in self.config.metrics.keys():
            if mode == "train":
                continue

            if "_" in mode:
                _mode, _submode = mode.split("_")
                if _mode != "eval":
                    raise ValueError("`mode` does not start with `eval_`")
                if not isinstance(self.config.env.eval, ConfigType):
                    raise ValueError(
                        f"""`self.config.env.eval` should either be a DictConfig.
                        Detected type is {type(self.config.env.eval)}"""
                    )
                if _submode in self.config.env.eval:
                    for _id in self.config.env[_mode][_submode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [_submode for _ in range(num_episodes_per_env)]
            elif mode == "eval":
                if isinstance(self.config.env.eval, ListConfigType):
                    for _id in self.config.env[mode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [mode for _ in range(num_episodes_per_env)]
            else:
                raise ValueError(f"eval mode = `{mode}` is not supported.")
        env = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )

        return env

    def run(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]
        print(env_indices)
        tmp_obs = copy.deepcopy(multitask_obs)
        train_mode = ["train" for _ in range(vec_env.num_envs)]
        success_goals = np.zeros((len(env_indices),20,3))
        pointer = np.zeros((len(env_indices)),dtype=np.int)
        pointer_cnt = np.zeros((len(env_indices)), dtype=np.int)
        num_tasks = len(env_indices)
        has_goals = np.zeros((len(env_indices)),dtype=np.int)
        current_use_pseudo_index = [False] * num_tasks
        pseudo_goals = np.zeros((num_tasks,3))

        success_cnt = np.zeros((len(env_indices), 20))
        multitask_pointer = 0
        success_set = []
        use_pseudo_idx = False
        pseudo_idx = 0
        success_sets = [[], [], []]
        pseudo_idxs = np.zeros([3], dtype=np.int)
        use_pseudo_idxs = [False, False, False]
        sets = [[], [], []]
        labels = np.zeros([num_tasks], dtype=np.int)
        print(multitask_obs['env_obs'][0])
        for i in range(multitask_obs['env_obs'].shape[0]):
            has_object_1 = not (torch.mean(multitask_obs['env_obs'][i,4:7])==0)
            has_object_2 = not (torch.mean(multitask_obs['env_obs'][i, 11:14]) == 0)
            if not has_object_1:
                sets[0].append(i)
                labels[i] = 0
            else:
                if not has_object_2:
                    sets[1].append(i)
                    labels[i] = 1
                else:
                    sets[2].append(i)
                    labels[i] = 2
        for i in range(3):
            print(sets[i],labels)
        success_eval = 0
        self.envs_to_exclude_during_training=range(20,30)
        self.envs_to_exclude_during_training =range(0,20)
        self.envs_to_exclude_during_training =[]
        # for i in range(20,30):
        #     self.envs_to_exclude_during_training.append(i)
        # for i in range(70,80):
        #     self.envs_to_exclude_during_training.append(i)
        # self.envs_to_exclude_during_training= [10,11,12,13,14,35,36,37,38,39]
        # self.envs_to_exclude_during_training = [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,40,41,42,43,44,45,46,47,48,49]
        # self.envs_to_exclude_during_training = range(10, 30)
        # self.envs_to_exclude_during_training = []
        use_expert = False
        expert_id = None
        relabel_id = None
        reuse = False
        reuse_id = 0
        ori = False
        use_expert = True
        expert_id = 1
        relabel_id = 2
        episode_samples = {'multitask_obs':[],"action":[],"reward":[],"next_multitask_obs":[],"done_bool":[],"task_obs":[],"info":[],"t":[],"expert_mean":[],"expert_var":[]} #actually log std, mu before tanh
        for step in range(self.start_step, exp_config.num_train_steps):

            if step % 300 == 0:  # todo
                if episode_samples['multitask_obs']!=[]:
                    newmtobs = copy.deepcopy(episode_samples['multitask_obs'][0])
                    newmtobs['env_obs'] = torch.cat([s['env_obs'] for s in episode_samples['multitask_obs']],0)
                    newmtobs['task_obs'] = torch.cat([s['task_obs'] for s in episode_samples['multitask_obs']],0)
                    expert_mean = np.zeros([newmtobs['env_obs'].shape[0],20,episode_samples['action'][0].shape[1]])
                    expert_var = np.zeros([newmtobs['env_obs'].shape[0], 20, episode_samples['action'][0].shape[1]])
                    # print(expert_mean.shape,len(episode_samples['multitask_obs']),)
                    if use_expert:
                        for i in range(3):
                            newmtobs['task_obs'] = torch.ones_like(newmtobs['task_obs'])*i
                            mean,var = self.agent.source_agents[i].act_all(newmtobs,modes=None,sample=False)
                            expert_mean[:,i,:] = mean
                            expert_var[:, i, :] = var
                        expert_mean[:,3:,:] = np.random.rand(expert_mean.shape[0],17,expert_mean.shape[2])*2-1
                        expert_var[:,3:,:] = -1.0
                        cnt = 3
                        # for i in [18, 44, 8, 31, 13, 49, 48, 23, 36, 6, 26, 38, 15, 21, 16, 35, 24, 37, 5, 20]:
                        # for i in range(3):
                        #     newmtobs['task_obs'] = torch.ones_like(newmtobs['task_obs']) * 0
                        #     mean, var = self.agent.source_agents[i+cnt].act_all(newmtobs, modes=None, sample=False)
                        #     expert_mean[:, i+cnt, :] = mean
                        #     expert_var[:, i+cnt, :] = var
                    # for i in [18, 44, 8]:
                    #     newmtobs['task_obs'] = torch.ones_like(newmtobs['task_obs'])*i
                    #     mean,var = self.agent.source_agents[-1].act_all(newmtobs,modes=None,sample=False)
                    #     expert_mean[:,cnt,:] = mean
                    #     expert_var[:, cnt, :] = var
                    #     cnt +=1
                    if step % 300 == 0:
                        print(expert_var[0,0],expert_var[0,1],expert_var[0,2],self.agent.kl_weight)
                    if '43_2'  in self.config.setup.load_dir:
                        expert_var = expert_var.clip(-2)
                    episode_samples["expert_mean"] = expert_mean
                    episode_samples["expert_var"] = expert_var

                    for s in range(len(episode_samples['multitask_obs'])):
                        for index, env_index in enumerate(env_indices):
                            self.replay_buffer.add(
                                episode_samples['multitask_obs'][s]["env_obs"][index],
                                episode_samples['action'][s][index],
                                episode_samples['reward'][s][index],
                                episode_samples['next_multitask_obs'][s]["env_obs"][index],
                                episode_samples['done_bool'][s][index],
                                task_obs=env_index,
                                info=episode_samples['info'][s][index]['grasp_success'],
                                t=episode_samples['t'][s][index],
                                expert_mean=episode_samples['expert_mean'][s*num_tasks+index],
                                expert_var=episode_samples['expert_var'][s*num_tasks+index]
                            )
                    episode_samples = {'multitask_obs': [], "action": [], "reward": [], "next_multitask_obs": [],
                                       "done_bool": [], "task_obs": [], "info": [], "t": [], "expert_mean": [],
                                       "expert_var": []}

            if step % self.max_episode_steps == 0:  # todo
                ori = True if np.random.rand()<=1.3 else False
                if step <= exp_config.init_steps*5:
                    ori=True
                if np.random.rand()<=0.05:
                    reuse = True
                    reuse_id = np.random.choice([0,1,2])
                else:
                    reuse = False
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")

                        success_cnt[:, multitask_pointer] = success
                        multitask_pointer = (multitask_pointer + 1) % (success_cnt.shape[1])
                        for i in range(success_cnt.shape[0]):
                            if i not in success_sets[labels[i]]:
                                if np.mean(success_cnt[i]) == 1:
                                    success_sets[labels[i]].append(i)
                        print('success set: ', success_sets)

                        for i in range(num_tasks):
                            if success[i]:
                                success_goals[i,pointer[i],:] = tmp_obs["env_obs"][i][-3:]
                                pointer_cnt[i] += 1
                                pointer[i] = (pointer[i]+1)%success_goals.shape[1]
                                has_goals[i] = 1
                        print('success set: ', pointer_cnt)
                        # if len(success_set) > 0:
                        #     idx = np.random.randint(0, len(success_set))
                        #     pseudo_idx = success_set[idx]
                        #     use_pseudo_idx = (np.random.rand() > 0.5)
                        #     print(pseudo_idx,use_pseudo_idx)
                        for index, _ in enumerate(env_indices):
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        self.logger.log("train/success", success.mean(), step)
                    for index, env_index in enumerate(env_indices):
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )
                        self.logger.log(f"train/env_index_{index}", env_index, step)

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    success_eval = self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode
                    )
                    # success_eval = self.plot_evaluate_vec_env_of_tasks(
                    #     vec_env=self.envs["eval"], step=step, episode=episode
                    # )
                    # self.hierarhichal_evaluate_vec_env_of_tasks(
                    #     vec_env=self.envs["eval"], step=step, episode=episode
                    # )
                    # if success_eval>(0.62-3):
                    #     self.envs_to_exclude_during_training=range(0,20)
                    # # if 0:
                    #     use_expert = True
                    #     expert_id = 1
                    #     relabel_id = 2

                    if exp_config.save.model:
                        self.agent.save(
                            self.model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )
                    # if success_eval > 0.9:
                    #     assert False
                print(self.envs_to_exclude_during_training,use_expert)
                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

            if step %  self.config.setup.pseudo_interval == 0:
                for m in range(num_tasks):
                    if has_goals[m]:
                        index = np.random.choice(min(pointer_cnt[m],success_goals.shape[1]))
                        pseudo_goals[m] = success_goals[m,index]
                        # distances = np.mean((success_goals[m,:min(pointer_cnt[m],success_goals.shape[1]),:]-multitask_obs['env_obs'][m][-3:].data.cpu().numpy())**2,axis=-1,keepdims=False)
                        # index = np.argmin(distances)
                        # pseudo_goals[m] = success_goals[m, index]
                    if np.random.rand()>self.config.setup.pseudo_thres and pointer_cnt[m]>10 and step > 0:
                        current_use_pseudo_index[m] = True
                        print(m,pointer_cnt[m],pseudo_goals[m])
                    else:
                        current_use_pseudo_index[m] = False

                for i in range(3):
                    if len(success_sets[i]) > 1:
                        idx = np.random.randint(0, len(success_sets[i]))
                        pseudo_idxs[i] = success_sets[i][idx]
                        use_pseudo_idxs[i] = (np.random.rand() > self.config.setup.pseudo_thres)
                        print(i, pseudo_idxs[i], use_pseudo_idxs[i])

            # print('!',multitask_obs['task_obs'])
            for i in range(self.config.setup.relabel_num_tasks):
                for j in range(self.config.setup.relabel_range):
                    multitask_obs['task_obs'][i * self.config.setup.relabel_range + j] = i
            # print('!!', multitask_obs['task_obs'])
            if step < exp_config.init_steps:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)

            else:
                with agent_utils.eval_mode(self.agent):
                    # multitask_obs = {"env_obs": obs, "task_obs": env_indices}
                    multitask_obs_copy = copy.deepcopy(multitask_obs)
                    # print(multitask_obs_copy['env_obs'].shape)
                    # for m in range(num_tasks):
                    #     if current_use_pseudo_index[m]:
                    #         multitask_obs_copy['env_obs'][m, -3:] = torch.FloatTensor(pseudo_goals[m]).to(multitask_obs_copy['env_obs'].device)
                    # for i in range(success_cnt.shape[0]):
                    #     if use_pseudo_idxs[labels[i]]:
                    #         if i not in success_sets[labels[i]]:
                    #             multitask_obs_copy['task_obs'][i] = pseudo_idxs[labels[i]]
                    #             multitask_obs_copy['env_obs'][i, -3:] = multitask_obs['env_obs'][i, -3:]
                    action = self.agent.sample_action(
                        multitask_obs=multitask_obs_copy,
                        modes=[
                            train_mode,
                        ],
                    )  # (num_envs, action_dim)
                    if not ori:
                        for i in range(num_tasks):
                            if np.random.rand()<=0.1:
                                reuse_id = np.random.choice([0,1,2])
                                multitask_obs_copy['task_obs'][i] = 0
                                action[i] = self.agent.source_agents[reuse_id].sample_action(
                                    multitask_obs=multitask_obs_copy,
                                    modes=[
                                        train_mode,
                                    ],
                                )[i]
                                # if reuse_id in [0,1]:
                                #     action[i] = self.agent.source_agent.sample_action(
                                #         multitask_obs=multitask_obs_copy,
                                #         modes=[
                                #             train_mode,
                                #         ],
                                #     )[i]
                                # else:
                                #     action[i] = self.agent.source_agent_2.sample_action(
                                #         multitask_obs=multitask_obs_copy,
                                #         modes=[
                                #             train_mode,
                                #         ],
                                #     )[i]
                        # if reuse:
                        #     for m in range(num_tasks):
                        #         multitask_obs_copy['task_obs'][m] = reuse_id
                        #     if reuse_id in [0,1]:
                        #         action = self.agent.source_agent.sample_action(
                        #             multitask_obs=multitask_obs_copy,
                        #             modes=[
                        #                 train_mode,
                        #             ],
                        #         )
                        #     else:
                        #         action = self.agent.source_agent_2.sample_action(
                        #             multitask_obs=multitask_obs_copy,
                        #             modes=[
                        #                 train_mode,
                        #             ],
                        #         )

            # run training update
            if step >= exp_config.init_steps:
                num_updates = (
                    exp_config.init_steps if step == exp_config.init_steps else 1
                )
                for _ in range(num_updates):

                # if step == exp_config.init_steps * 10:
                #     self.agent.update(self.replay_buffer, self.logger, step, use_expert=use_expert, expert_id=expert_id,
                #                       relabel_id=relabel_id, sync=True, use_pig=True)
                # elif step > exp_config.init_steps * 10:
                #     self.agent.update(self.replay_buffer, self.logger, step, use_expert=use_expert, expert_id=expert_id,
                #                       relabel_id=relabel_id, sync=True, use_pig=True)
                # else:
                #     self.agent.update(self.replay_buffer, self.logger, step, use_expert=False, expert_id=None,
                #                       relabel_id=None)

                #     if success_eval <=0.6:
                #         use_expert = False
                #         expert_id = None
                #         relabel_id = None
                #     else:
                #         use_expert = True
                #         expert_id = 1
                #         relabel_id = 2
                    if step >= exp_config.init_steps*3:
                        self.agent.update(self.replay_buffer, self.logger, step,use_expert=use_expert,expert_id=expert_id,relabel_id=relabel_id)
                    else:
                        self.agent.update(self.replay_buffer, self.logger, step, use_expert=False, expert_id=None,
                                          relabel_id=None)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
                    tmp_obs = copy.deepcopy(multitask_obs)
            # print('!', next_multitask_obs['task_obs'])
            for i in range(self.config.setup.relabel_num_tasks):
                for j in range(self.config.setup.relabel_range):
                    next_multitask_obs['task_obs'][i*self.config.setup.relabel_range+j] = i
            # if self.config.setup.relabel_num_tasks==3:
            #     for j in range(self.config.setup.relabel_range):
            #         reward[0 + j] = reward[0 + j] if info[0+j]['success'] else 0
            #         reward[10 + j] = reward[10 + j] if info[10 + j]['success'] else 0
            #         reward[20 + j] = reward[20 + j] if info[20 + j]['success'] else 0
            # for i in range(self.config.setup.relabel_num_tasks):
            #     for j in range(self.config.setup.relabel_range):
            #         reward[i*self.config.setup.relabel_range+j] = reward[i*self.config.setup.relabel_range+j] if info[i*self.config.setup.relabel_range+j]['success'] else 0
            # elif self.config.setup.relabel_num_tasks == 1:
            #     for j in range(self.config.setup.relabel_range):
            #         reward[0 + j] = reward[0 + j] if info[0 + j]['success'] else 0
            # elif self.config.setup.relabel_num_tasks==3:
            #     for j in range(self.config.setup.relabel_range):
            #         reward[20 + j] = reward[20 + j] if info[20+j]['success'] else 0
            # print('!!', next_multitask_obs['task_obs'])
            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += np.asarray([x["success"] for x in info])

            # allow infinite bootstrap
            # episode_samples = {'multitask_obs': [], "action": [], "reward": [], "next_multitask_obs": [],
            #                    "done_bool": [], "task_obs": [], "info": [], "t": [], "expert_mean": [],
            #                    "expert_var": []}
            episode_samples['multitask_obs'].append(multitask_obs)
            episode_samples['action'].append(action)
            episode_samples['reward'].append(reward)
            episode_samples['next_multitask_obs'].append(next_multitask_obs)
            done_bool = []
            t = []
            for index, env_index in enumerate(env_indices):
                done_bool.append(
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                t.append(episode_step[index] % self.max_episode_steps)

            episode_samples['done_bool'].append(np.array(done_bool))
            episode_samples['info'].append(info)
            episode_samples['t'].append(np.array(t))

            # for index, env_index in enumerate(env_indices):
            #     done_bool = (
            #         0
            #         if episode_step[index] + 1 == self.max_episode_steps
            #         else float(done[index])
            #     )
            #     if index not in self.envs_to_exclude_during_training:
            #         # print(env_index,multitask_obs["task_obs"][index],env_indices)
            #         self.replay_buffer.add(
            #             multitask_obs["env_obs"][index],
            #             action[index],
            #             reward[index],
            #             next_multitask_obs["env_obs"][index],
            #             done_bool,
            #             task_obs=env_index,
            #             info=info[index]['grasp_success'],
            #             t=episode_step[index]%self.max_episode_steps
            #         )

            multitask_obs = next_multitask_obs
            episode_step += 1
        self.replay_buffer.delete_from_filesystem(self.buffer_dir)
        self.close_envs()

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        """Collect some trajectories, by unrolling the policy (in train mode),
        and update the replay buffer.
        Args:
            vec_env (VecEnv): environment to collect data from.
            num_steps (int): number of steps to collect data for.

        """
        raise NotImplementedError
