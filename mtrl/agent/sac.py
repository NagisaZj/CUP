# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
import copy
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from mtrl.agent import utils as agent_utils
from mtrl.agent.components.actor import _gaussian_logprob, _squash
from mtrl.agent.abstract import Agent as AbstractAgent
from mtrl.agent.ds.mt_obs import MTObs, InfoMTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.utils.types import ConfigType, ModelType, ParameterType, TensorType

def gaussian_kld(
    mean1: TensorType, logvar1: TensorType, mean2: TensorType, logvar2: TensorType
) -> TensorType:
    """Compute KL divergence between a bunch of univariate Gaussian
        distributions with the given means and log-variances.
        ie `KL(N(mean1, logvar1) || N(mean2, logvar2))`

    Args:
        mean1 (TensorType):
        logvar1 (TensorType):
        mean2 (TensorType):
        logvar2 (TensorType):

    Returns:
        TensorType: [description]
    """

    gauss_klds = 0.5 * (
        (logvar2 - logvar1)
        + ((torch.exp(logvar1) + (mean1 - mean2) ** 2.0) / torch.exp(logvar2))
        - 1.0
    )
    assert len(gauss_klds.size()) == 2
    return gauss_klds

class Agent(AbstractAgent):
    """SAC algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
            # use_expert=True
    ):
        self.idx = None
        self.significance = None
        self.cnt = 0
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        self.idx = None
        self.significance = None
        self.cnt = 0
        self.should_use_task_encoder = self.multitask_cfg.should_use_task_encoder
        self.kl_weight = 30
        self.clip_thres = 3e-3

        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.actor = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(init_temperature, dtype=np.float32)
                    for _ in range(self.num_envs)
                ]
            ).to(self.device)
        )
        # self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self._components = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_alpha": self.log_alpha,  # type: ignore[dict-item]
        }
        # optimizers
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )
        if loss_reduction not in ["mean", "none"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )

        self.loss_reduction = loss_reduction

        self._optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "log_alpha": self.log_alpha_optimizer,
        }

        if self.should_use_task_encoder:
            self.task_encoder = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.model_cfg,
            ).to(self.device)
            name = "task_encoder"
            self._components[name] = self.task_encoder
            self.task_encoder_optimizer = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.optimizer_cfg,
                params=self.get_parameters(name=name),
            )
            self._optimizers[name] = self.task_encoder_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        if cfg_to_load_model:
            self.load(**cfg_to_load_model)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name != "log_alpha":
                component.train(training)

    def get_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.multitask_cfg.should_use_disentangled_alpha:
            return self.log_alpha[env_index].exp()
        else:
            return self.log_alpha[0].exp()

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(env_index.to(self.device))
        return self.task_encoder(env_index.to(self.device))

    def act(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            # assert action.ndim == 2 and action.shape[0] == 1
            return action.detach().cpu().numpy()

    def act_all(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ):
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, log_std,old_mu = self.actor.forward_ori_mu(mtobs=mtobs)
            # assert action.ndim == 2 and action.shape[0] == 1
            return old_mu.detach().cpu().numpy(), log_std.detach().cpu().numpy(),

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=False)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=True)

    def get_mean_var(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act_all(multitask_obs=multitask_obs, modes=modes, sample=False)

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")

    def _compute_gradient(
        self,
        loss: TensorType,
        parameters: List[ParameterType],
        step: int,
        component_names: List[str],
        retain_graph: bool = False,
    ):
        """Method to override the gradient computation.

            Useful for algorithms like PCGrad and GradNorm.

        Args:
            loss (TensorType):
            parameters (List[ParameterType]):
            step (int): step for tracking the training of the agent.
            component_names (List[str]):
            retain_graph (bool, optional): if it should retain graph. Defaults to False.
        """
        loss.backward(retain_graph=retain_graph)

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        mu, policy_action, log_pi, log_std = self.actor(mtobs=mtobs)
        # mu, pi, log_pi, log_std, old_mu = self.actor.my_forward(mtobs=mtobs, detach_encoder=True)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=policy_action)
        # target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=pi, detach_encoder=True)
        Q = torch.min(target_Q1, target_Q2)#.detach()
        # min_Q = torch.min(target_Q1, target_Q2).detach()

        # Q = Q / 3
        Q = Q - self.get_alpha(batch.task_obs).detach() * log_pi
        # print(target_Q1.shape,self.get_alpha(batch.task_obs).detach().shape,log_pi.shape,torch.sum(self.get_alpha(batch.task_obs).detach() * log_pi,-1,keepdim=True).shape)
        # use_expert

        # if 1:
        #     means = batch.expert_means
        #     vars = batch.expert_vars
        #     for i in range(3):
        #         Q_e = torch.zeros_like(Q)
        #         std_e = vars[:,i,:].exp()
        #         for j in range(1):
        #             noise = torch.randn_like(means[:,i,:])
        #             pi_e_n = means[:,i,:] + noise * std_e
        #             log_pi_e = _gaussian_logprob(noise, vars[:,i,:])
        #             new_mean, pi_e_n, _ = _squash(means[:,i,:], pi_e_n, None)
        #             Q_e_1_n, Q_e_2_n = self.critic_target(mtobs=mtobs, action=pi_e_n, detach_encoder=True)
        #             Q_e = (torch.min(Q_e_1_n, Q_e_2_n)) - self.get_alpha(batch.task_obs).detach() * log_pi_e
        #         Q = torch.max(Q,Q_e)
        return Q

    def update_critic(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        with torch.no_grad():
            target_V = self._get_target_V(batch=batch, task_info=task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        current_Q1, current_Q2 = self.critic(
            mtobs=mtobs,
            action=batch.action,
            detach_encoder=False,
        )
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            raise RuntimeError(
                f"critic_loss = {loss_to_log} is too high. Stopping training."
            )

        component_names = ["critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="critic"),1)
        # Optimize the critic
        self.critic_optimizer.step()



    def get_weights(
        self,
        multitask_obs: ObsType    ):
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        modes=['eval']
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            action=mu
            expert_actions = []
            Q_s = []
            Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu, detach_encoder=True)
            Q_e = torch.max(Q_e_1, Q_e_2).detach()
            Q_s.append(Q_e.detach().cpu().numpy()[0])
            for i in [0,1]:
                mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agents[i].actor.forward_expert_auto(
                    mtobs=mtobs, detach_encoder=True,
                    expert_id=i,
                    relabel_id=None)
                expert_actions.append(mu_e)
                Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu_e, detach_encoder=True)
                Q_e = torch.max(Q_e_1, Q_e_2).detach()
                Q_s.append(Q_e.detach().cpu().numpy()[0])
            for i in [2]:
                mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agents[i].actor.forward_expert_auto(
                    mtobs=mtobs, detach_encoder=True,
                    expert_id=i,
                    relabel_id=None)
                expert_actions.append(mu_e)
                Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu_e, detach_encoder=True)
                Q_e = torch.max(Q_e_1, Q_e_2).detach()
                Q_s.append(Q_e.detach().cpu().numpy()[0])
            # assert action.ndim == 2 and action.shape[0] == 1
            return Q_s

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
        use_expert=False,
        expert_id=None,
        relabel_id=None
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """

        # detach encoder, so we don't update it with the actor loss
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info
        )
        infomtobs = InfoMTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info,
            info=batch.infos,
            t=batch.ts
        )
        mu, pi, log_pi, log_std, old_mu = self.actor.my_forward(mtobs=mtobs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(mtobs=mtobs, action=pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.loss_reduction == "mean":
            actor_loss = (
                self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            ).mean()
            logger.log("train/actor_loss", actor_loss, step)

        elif self.loss_reduction == "none":
            actor_loss = self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            logger.log("train/actor_loss", actor_loss.mean(), step)



        # if use_expert:
        #     num_experts = 3
        #     num_samples = 3
        #     bs = pi.shape[0]
        #     action_list = torch.ones_like(pi).repeat(num_samples*(num_experts+1), 1)
        #     newtaskinfo = TaskInfo(encoding=task_info.encoding,compute_grad=task_info.compute_grad,env_index=task_info.env_index.repeat((num_experts+1)*num_samples,1))
        #     newmtobs = MTObs(
        #         env_obs=batch.env_obs.repeat(num_samples*(num_experts+1), 1),
        #         task_obs=None,
        #         task_info=newtaskinfo
        #     )
        #     action_list[(num_samples*num_experts)*bs:(num_samples*num_experts+1)*bs,:] = pi
        #     std = log_std.exp()
        #     for j in range(num_samples-1):
        #         noise = torch.randn_like(old_mu)
        #         pi_n = old_mu + noise * std
        #         _, pi_n, _ = _squash(old_mu, pi_n, None)
        #         action_list[(num_samples*num_experts+j+1)*bs:(num_samples*num_experts+j+2)*bs, :] = pi_n
        #     mu_e_all, pi_e_all, log_pi_e_all, log_std_e_all,Q_e_all,kld_all = [mu.detach()], [pi.detach()], [log_pi.detach()], [log_std.detach()], [],[]
        #
        #     source_list = [18, 13, 49, 44, 8, 20, 24, 48, 21, 36, 23, 6, 26, 31, 25, 37, 14, 16, 15, 38, 5, 4, 7, 19, 35]
        #     # candids = np.random.choice(source_list,3,replace=False)
        #     means = batch.expert_means
        #     vars = batch.expert_vars
        #     for i in [0,1,2]:
        #         std_e = vars[:,i,:].exp()
        #         for j in range(3):
        #             noise = torch.randn_like(means[:,i,:])
        #             pi_e_n = means[:,i,:] + noise * std_e
        #             new_mean, pi_e_n, _ = _squash(means[:,i,:], pi_e_n, None)
        #             action_list[(i * num_samples+j) * bs:(i * num_samples + 1+j) * bs, :] = pi_e_n
        #         kld_tmp = gaussian_kld(mu, log_std * 1, new_mean, vars[:,i,:] * 1)
        #         kld_tmp = torch.mean(kld_tmp, 1, keepdim=True)
        #         # print(kld_tmp.shape)
        #         kld_all.append(kld_tmp)
        #
        #
        #     Q_e_1,Q_e_2 = self.critic_target(mtobs=newmtobs, action=action_list, detach_encoder=True)
        #     Q_max = torch.max(Q_e_1,Q_e_2).detach()
        #     Q_max = Q_max.reshape(num_samples,num_experts+1,bs,-1)
        #     Q_max = Q_max.mean(0)
        #     Q_e_all = torch.zeros_like(Q_max).reshape(bs,num_experts+1)
        #     for j in range(num_experts+1):
        #         Q_e_all[:,j:(j+1)] = Q_max[j]
        #
        #     kld_all.append(torch.zeros_like(kld_all[0]))
        #     # Q_e_all = torch.cat(Q_e_all,1)
        # #     policy_entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
        # #     dim=-1
        # # )
        #     # weight = torch.softmax(Q_e_all,1).detach()
        #     # # print(torch.mean(weight,0))
        #     #
        #     # kld = None
        #     # for i in range(1, 3):
        #     #     if kld == None:
        #     #         kld = kld_all[i] * weight[:, i:(i + 1)]
        #     #     else:
        #     #         kld += kld_all[i] * weight[:, i:(i + 1)]
        #     #
        #     # kld_loss = (kld * mask).mean()
        #     # if kld_loss > 10:
        #     #     kld_loss = kld_loss/(kld_loss.detach())
        #
        #     num_candidates = Q_e_all.shape[1]
        #
        #
        #     kld_all = torch.cat(kld_all,1)
        #     # weight = torch.softmax(Q_e_all,1).detach()
        #     # print(weight.shape,kld_all.shape)
        #     x,y = torch.max(Q_e_all,1)
        #     weight = torch.nn.functional.one_hot(y,num_classes=num_candidates)
        #     Q_weight = Q_e_all - Q_e_all[:,-1:]
        #     kld_loss = (kld_all * weight*Q_weight).mean()
        #     if kld_loss > 100:
        #         kld_loss = kld_loss/(kld_loss.detach())*100
        #     self.cnt = (self.cnt + 1) % 10000
        #
        #     mean_weight = torch.mean(weight.float(), 0)
        #     if self.cnt == 1:
        #         print(mean_weight,weight[0],torch.mean(kld_all,0),[s.mean() for s in log_std_e_all])
        #         for j in range(1,num_candidates):
        #             logger.log("train/weight_%d"%(j), mean_weight[j-1], step)
        #         logger.log("train/weight_%d" % (0), mean_weight[-1], step)


        if use_expert:
            target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=pi, detach_encoder=True)
            # Q1, Q2 = target_Q1.clone(), target_Q2.clone()
            Q = torch.max(target_Q1, target_Q2).detach()
            Q = Q -self.get_alpha(batch.task_obs).detach() * log_pi
            # min_Q = torch.min(target_Q1, target_Q2).detach()
            min_Q = ((target_Q1 + target_Q2) / 2).detach()
            std = log_std.exp()
            for j in range(2):
                noise = torch.randn_like(old_mu)
                pi_n = old_mu + noise * std
                _, pi_n, _ = _squash(old_mu, pi_n, None)
                Q_1_n, Q_2_n = self.critic_target(mtobs=mtobs, action=pi_n, detach_encoder=True)
                Q = Q + (torch.max(Q_1_n, Q_2_n)).detach()
                # Q1 += Q_1_n
                # Q2 += Q_2_n
                # min_Q = min_Q + (torch.min(Q_1_n, Q_2_n)).detach()
                min_Q = min_Q + ((Q_1_n + Q_2_n) / 2).detach()
                log_pi_e = _gaussian_logprob(noise, log_std)
                Q = Q - self.get_alpha(batch.task_obs).detach() * log_pi_e#


            Q = Q / 3
            min_Q = min_Q / 3
            # Q = Q - torch.sum(self.get_alpha(batch.task_obs).detach() * log_std,-1,keepdim=True)
            # Q1 = Q1 / 3
            # Q2 = Q2 / 3
            # t_q = torch.min(Q1, Q2)- self.get_alpha(env_index=batch.task_obs).detach() * log_pi
            # t_q = t_q.detach()
            mu_e_all, pi_e_all, log_pi_e_all, log_std_e_all,Q_e_all,kld_all = [mu.detach()], [pi.detach()], [log_pi.detach()], [log_std.detach()], [Q],[None]
            min_Q_all = [min_Q]
            means = batch.expert_means
            vars = batch.expert_vars
            for i in range(3):
                Q_e = torch.zeros_like(Q)
                mi_Q_e = torch.zeros_like(Q)
                std_e = vars[:,i,:].exp()
                for j in range(3):
                    noise = torch.randn_like(means[:,i,:])
                    pi_e_n = means[:,i,:] + noise * std_e
                    new_mean, pi_e_n, _ = _squash(means[:,i,:], pi_e_n, None)
                    log_pi_e = _gaussian_logprob(noise, vars[:,i,:])
                    Q_e_1_n, Q_e_2_n = self.critic_target(mtobs=mtobs, action=pi_e_n, detach_encoder=True)
                    Q_e = Q_e + (torch.max(Q_e_1_n, Q_e_2_n)).detach()
                    Q_e = Q_e - self.get_alpha(batch.task_obs).detach() * log_pi_e
                    # mi_Q_e = mi_Q_e + (torch.min(Q_e_1_n, Q_e_2_n)).detach()
                    mi_Q_e = mi_Q_e + ((Q_e_1_n+ Q_e_2_n)/2).detach()
                Q_e = Q_e / 3
                mi_Q_e = mi_Q_e / 3
                # Q_e = Q_e - torch.sum(self.get_alpha(batch.task_obs).detach() * vars[:,i,:], -1, keepdim=True)
                Q_e_all.append(Q_e)
                min_Q_all.append(mi_Q_e)
                kld_tmp = gaussian_kld(mu, log_std * 1, new_mean, vars[:,i,:] * 1)
                kld_tmp = torch.mean(kld_tmp, 1, keepdim=True)
                # print(kld_tmp.shape)
                kld_all.append(kld_tmp)
            kld_all[0] = torch.zeros_like(kld_all[-1])
            Q_e_all = torch.cat(Q_e_all, 1)
            min_Q_all = torch.cat(min_Q_all, 1)

            num_candidates = Q_e_all.shape[1]


            kld_all = torch.cat(kld_all,1)
            # weight = torch.softmax(Q_e_all,1).detach()
            # print(weight.shape,kld_all.shape)
            x,y = torch.max(Q_e_all,1)
            # x, y = torch.max(min_Q_all, 1)
            weight = torch.nn.functional.one_hot(y,num_classes=num_candidates)
            # Q_weight = min_Q_all.detach() - min_Q_all[:, 0:1].detach()
            # Q_weight = Q_weight.clamp(0, 100)
            Q_weight = Q_e_all.detach() - Q_e_all[:,0:1].detach()
            Q_weight = torch.min(Q_weight,abs(Q_e_all[:, 0:1]*self.clip_thres))
            Q_weight = Q_weight.detach()
            # mean_Q_weight = Q_weight.detach()
            # if mean_Q_weight > 0.1:
            #     Q_weight = Q_weight / mean_Q_weight * 0.1
            mean_Q_weight = (weight * Q_weight).sum(1).mean()
            kld_loss = (kld_all * weight*Q_weight).mean()
            # if kld_loss > 100:
            #     kld_loss = kld_loss/(kld_loss.detach())*100
            self.cnt = (self.cnt + 1) % 10000

            mean_weight = torch.mean(weight.float(), 0)
            if self.cnt == 1:
                print(mean_weight,weight[0],torch.mean(kld_all,0),[s.mean() for s in log_std_e_all])
                print(mean_Q_weight)
                print(Q_e_all[:,0:1].mean())
                for j in range(num_candidates):
                    logger.log("train/weight_%d"%j, mean_weight[j], step)


            # self.cnt = (self.cnt+1)%10000
            # time_masks = []
            # time_Q_means = torch.zeros([5,num_candidates]).to(infomtobs.task_info.env_index.device)
            # for i in range(5):
            #     time_masks.append(((infomtobs.t >= i*100)*(infomtobs.t < (i+1)*100)).float().to(infomtobs.task_info.env_index.device).detach())
            #     if self.idx ==None or self.cnt ==0:
            #
            #         for j in range(num_candidates):
            #             # print(Q_e_all.shape,time_masks[i].shape)
            #             time_Q_means[i,j] = torch.mean(Q_e_all[:,j:(j+1)]*time_masks[i]*mask).detach()
            #             # if j==0:
            #             #     time_Q_means[i, j] -= 1e-5
            # if self.idx == None or self.cnt == 0:
            #     # indexs = torch.argmax(time_Q_means[:,1:],1)
            #     indexs = torch.softmax(time_Q_means,1)
            #     expected_improvement = torch.max(indexs,1)[0]  - torch.min(indexs,1)[0]
            #     self.significance = (expected_improvement*num_candidates).mean()
            #     self.significance = torch.clamp(self.significance,0.5,0.5).detach()*2
            #     self.idx = indexs.detach()
            #     print(self.idx)
            #     print([s.mean() for s in log_std_e_all],[s.mean() for s in kld_all])
            #     print(self.significance)
            #     for j in range(num_candidates):
            #         logger.log("train/weight_%d"%j, self.idx[:,j].mean(), step)
            #     logger.log('train/significance',self.significance.mean(),step)
            # # print(indexs)
            # kld = None
            # for i in range(5):
            #     for j in range(num_candidates):
            #         if kld ==None:
            #             kld = kld_all[j]*time_masks[i]*self.idx[i,j]
            #         else:
            #             kld += kld_all[j]*time_masks[i]*self.idx[i,j]
            #
            # kld_loss = (kld * mask*self.significance).mean()
            # if kld_loss > 100:
            #     kld_loss = kld_loss/(kld_loss.detach())*100





            # kld_loss = kld_loss / (kld_loss.detach())
            # actor_loss_scale = (torch.ones_like(actor_loss)*actor_loss).detach()
            # actor_loss_scale = torch.abs(actor_loss_scale)
            # kld_loss = kld_loss * (actor_loss_scale.detach()) * (torch.sum(mean_weight[1:]).detach())
            kld_loss = kld_loss*self.kl_weight
            actor_loss+=kld_loss
            # actor_loss-=policy_entropy.mean()*(3e-1)*(1-self.significance)
            logger.log("train/expert_kl", kld_loss, step)
            logger.log("train/Q_weight",mean_Q_weight,step)




        logger.log("train/actor_target_entropy", self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )

        logger.log("train/actor_entropy", entropy.mean(), step)

        # optimize the actor
        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="actor"), 1)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        if self.loss_reduction == "mean":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            ).mean()
            logger.log("train/alpha_loss", alpha_loss, step)
        elif self.loss_reduction == "none":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            )
            logger.log("train/alpha_loss", alpha_loss.mean(), step)
        # breakpoint()
        # logger.log("train/alpha_value", self.get_alpha(batch.task_obs), step)
        self._compute_gradient(
            loss=alpha_loss,
            parameters=self.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="log_alpha"), 1)
        self.log_alpha_optimizer.step()

    def get_task_info(
        self, task_encoding: TensorType, component_name: str, env_index: TensorType
    ) -> TaskInfo:
        """Encode task encoding into task info.

        Args:
            task_encoding (TensorType): encoding of the task.
            component_name (str): name of the component.
            env_index (TensorType): index of the environment.

        Returns:
            TaskInfo: TaskInfo object.
        """
        if self.should_use_task_encoder:
            if component_name in self.multitask_cfg.task_encoder_cfg.losses_to_train:
                task_info = TaskInfo(
                    encoding=task_encoding, compute_grad=True, env_index=env_index
                )
            else:
                task_info = TaskInfo(
                    encoding=task_encoding.detach(),
                    compute_grad=False,
                    env_index=env_index,
                )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info

    def update_transition_reward_model(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the transition model and reward decoder.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the task encoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        self.task_encoder_optimizer.step()

    def update_decoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the decoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
        use_expert=False,
        expert_id=None,
        relabel_id=None
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """

        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if buffer_index_to_sample is None:
            batch = replay_buffer.sample()
        else:
            batch = replay_buffer.sample(buffer_index_to_sample)

        logger.log("train/batch_reward", batch.reward.mean(), step)
        if self.should_use_task_encoder:
            self.task_encoder_optimizer.zero_grad()
            task_encoding = self.get_task_encoding(
                env_index=batch.task_obs.squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            task_encoding = None  # type: ignore[assignment]

        task_info = self.get_task_info(
            task_encoding=task_encoding,
            component_name="critic",
            env_index=batch.task_obs,
        )
        self.update_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )
        if step % self.actor_update_freq == 0:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="actor",
                env_index=batch.task_obs,
            )
            self.update_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
                use_expert=use_expert,
                expert_id=expert_id,
                relabel_id=relabel_id
            )
        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if (
            "transition_model" in self._components
            and "reward_decoder" in self._components
        ):
            # some of the logic is a bit sketchy here. We will get to it soon.
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="transition_reward",
                env_index=batch.task_obs,
            )
            self.update_transition_reward_model(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if (
            "decoder" in self._components  # should_update_decoder
            and self.decoder is not None  # type: ignore[attr-defined]
            and step % self.decoder_update_freq == 0  # type: ignore[attr-defined]
        ):
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="decoder",
                env_index=batch.task_obs,
            )
            self.update_decoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        if self.should_use_task_encoder:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="task_encoder",
                env_index=batch.task_obs,
            )
            self.update_task_encoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        return batch.buffer_index

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]:
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            # print(list(self.actor.model.parameters()))
            # torch.Size([400, 39])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([8, 400])
            # torch.Size([8])

            # torch.Size([1, 39, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 8])
            # torch.Size([1, 1, 8])

            # torch.Size([400, 43])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([1, 400])
            # torch.Size([1])

            # torch.Size([400, 43])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([400, 400])
            # torch.Size([400])
            # torch.Size([1, 400])
            # torch.Size([1])


            # torch.Size([1, 43, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 1])
            # torch.Size([1, 1, 1])

            # torch.Size([1, 43, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 400])
            # torch.Size([1, 1, 400])
            # torch.Size([1, 400, 1])
            # torch.Size([1, 1, 1])



            # for p in self._components['actor'].parameters():
            #     print(p.shape)
            return list(self.actor.model.parameters())
        elif name in ["log_alpha", "alpha"]:
            return [self.log_alpha]
        elif name == "encoder":
            return list(self.critic.encoder.parameters())
        else:
            return list(self._components[name].parameters())


class ExpertAgent(AbstractAgent):
    """SAC algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
            # use_expert=True
    ):
        self.idx = None
        self.significance = None
        self.cnt = 0
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        self.idx = None
        self.significance = None
        self.cnt = 0
        self.should_use_task_encoder = self.multitask_cfg.should_use_task_encoder

        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.actor = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.pig_critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.pig_critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(init_temperature, dtype=np.float32)
                    for _ in range(self.num_envs)
                ]
            ).to(self.device)
        )
        # self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self._components = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_alpha": self.log_alpha,  # type: ignore[dict-item]
            "pig_critic": self.pig_critic,
            "pig_critic_target": self.pig_critic_target,
        }
        # optimizers
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.pig_critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="pig_critic")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )
        if loss_reduction not in ["mean", "none"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )

        self.loss_reduction = loss_reduction

        self._optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "log_alpha": self.log_alpha_optimizer,
            "pig_critic": self.pig_critic_optimizer,
        }

        if self.should_use_task_encoder:
            self.task_encoder = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.model_cfg,
            ).to(self.device)
            name = "task_encoder"
            self._components[name] = self.task_encoder
            self.task_encoder_optimizer = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.optimizer_cfg,
                params=self.get_parameters(name=name),
            )
            self._optimizers[name] = self.task_encoder_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        if cfg_to_load_model:
            self.load(**cfg_to_load_model)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.pig_critic.load_state_dict(self.critic.state_dict())
        self.pig_critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name != "log_alpha":
                component.train(training)

    def get_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.multitask_cfg.should_use_disentangled_alpha:
            return self.log_alpha[env_index].exp()
        else:
            return self.log_alpha[0].exp()

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(env_index.to(self.device))
        return self.task_encoder(env_index.to(self.device))

    def act(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            # assert action.ndim == 2 and action.shape[0] == 1
            return action.detach().cpu().numpy()

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=False)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=True)

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")

    def _compute_gradient(
        self,
        loss: TensorType,
        parameters: List[ParameterType],
        step: int,
        component_names: List[str],
        retain_graph: bool = False,
    ):
        """Method to override the gradient computation.

            Useful for algorithms like PCGrad and GradNorm.

        Args:
            loss (TensorType):
            parameters (List[ParameterType]):
            step (int): step for tracking the training of the agent.
            component_names (List[str]):
            retain_graph (bool, optional): if it should retain graph. Defaults to False.
        """
        loss.backward(retain_graph=retain_graph)

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        _, policy_action, log_pi, _ = self.actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=policy_action)
        return (
            torch.min(target_Q1, target_Q2)
            - self.get_alpha(env_index=batch.task_obs).detach() * log_pi
        )

    def _get_target_pig_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        _, policy_action, log_pi, _ = self.actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=policy_action)
        target_pig_Q1, target_pig_Q2 = self.pig_critic_target(mtobs=mtobs, action=policy_action)
        min_Q = torch.max(target_Q1, target_Q2)
        min_pig_Q = torch.min(target_pig_Q1, target_pig_Q2)
        Q_mat = min_Q
        pig_Q_mat = min_pig_Q
        # target_Q = min_Q
        infomtobs = InfoMTObs(
            env_obs=batch.next_env_obs,
            task_obs=None,
            task_info=task_info,
            info=batch.infos,
            t=batch.ts
        )
        for i in [0,1]:
            mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agent.actor.forward_expert_auto(mtobs=infomtobs,
                                                                                                        detach_encoder=True,
                                                                                                        expert_id=i,
                                                                                                        relabel_id=None)
            target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=pi_e)
            target_pig_Q1, target_pig_Q2 = self.pig_critic_target(mtobs=mtobs, action=policy_action)
            min_Q = torch.max(target_Q1, target_Q2)
            min_pig_Q = torch.min(target_pig_Q1, target_pig_Q2)
            Q_mat = torch.cat([Q_mat,min_Q],1)
            pig_Q_mat = torch.cat([pig_Q_mat,min_pig_Q],1)
        for i in [2]:
            mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agent_2.actor.forward_expert_auto(mtobs=infomtobs,
                                                                                                        detach_encoder=True,
                                                                                                        expert_id=i,
                                                                                                        relabel_id=None)
            target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=pi_e)
            target_pig_Q1, target_pig_Q2 = self.pig_critic_target(mtobs=mtobs, action=policy_action)
            min_Q = torch.max(target_Q1, target_Q2)
            min_pig_Q = torch.min(target_pig_Q1, target_pig_Q2)
            Q_mat = torch.cat([Q_mat, min_Q], 1)
            pig_Q_mat = torch.cat([pig_Q_mat, min_pig_Q], 1)
        x, y = torch.max(Q_mat, 1)
        weight = torch.nn.functional.one_hot(y, num_classes=Q_mat.shape[1])
        target_Q = torch.sum(pig_Q_mat * weight,1,keepdim=True)
        return target_Q

    def update_critic(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        with torch.no_grad():
            target_V = self._get_target_V(batch=batch, task_info=task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        current_Q1, current_Q2 = self.critic(
            mtobs=mtobs,
            action=batch.action,
            detach_encoder=False,
        )
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            raise RuntimeError(
                f"critic_loss = {loss_to_log} is too high. Stopping training."
            )

        component_names = ["critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="critic"),1)
        # Optimize the critic
        self.critic_optimizer.step()

    def update_pig_critic(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        with torch.no_grad():
            target_V = self._get_target_pig_V(batch=batch, task_info=task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        current_Q1, current_Q2 = self.pig_critic(
            mtobs=mtobs,
            action=batch.action,
            detach_encoder=False,
        )
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/pig_critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            raise RuntimeError(
                f"pig_critic_loss = {loss_to_log} is too high. Stopping training."
            )

        component_names = ["pig_critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="critic"),1)
        # Optimize the critic
        self.critic_optimizer.step()

    def get_weights(
        self,
        multitask_obs: ObsType    ):
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        modes=['eval']
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            action=mu
            expert_actions = []
            Q_s = []
            Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu, detach_encoder=True)
            Q_e = torch.max(Q_e_1, Q_e_2).detach()
            Q_s.append(Q_e.detach().cpu().numpy()[0])
            for i in [0,1]:
                mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agent.actor.forward_expert_auto(
                    mtobs=mtobs, detach_encoder=True,
                    expert_id=i,
                    relabel_id=None)
                expert_actions.append(mu_e)
                Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu_e, detach_encoder=True)
                Q_e = torch.max(Q_e_1, Q_e_2).detach()
                Q_s.append(Q_e.detach().cpu().numpy()[0])
            for i in [2]:
                mu_e, pi_e, log_pi_e, log_std_e, mask, old_mu = self.source_agent_2.actor.forward_expert_auto(
                    mtobs=mtobs, detach_encoder=True,
                    expert_id=i,
                    relabel_id=None)
                expert_actions.append(mu_e)
                Q_e_1, Q_e_2 = self.critic_target(mtobs=mtobs, action=mu_e, detach_encoder=True)
                Q_e = torch.max(Q_e_1, Q_e_2).detach()
                Q_s.append(Q_e.detach().cpu().numpy()[0])
            # assert action.ndim == 2 and action.shape[0] == 1
            return Q_s

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
        use_expert=False,
        expert_id=None,
        relabel_id=None,
        use_pig=False
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """

        # detach encoder, so we don't update it with the actor loss
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info
        )
        mu, pi, log_pi, log_std, old_mu = self.actor.my_forward(mtobs=mtobs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(mtobs=mtobs, action=pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.loss_reduction == "mean":
            actor_loss = (
                self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            ).mean()
            logger.log("train/actor_loss", actor_loss, step)

        elif self.loss_reduction == "none":
            actor_loss = self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            logger.log("train/actor_loss", actor_loss.mean(), step)
        if use_pig:
            actor_pig_Q1, actor_pig_Q2 = self.pig_critic(mtobs=mtobs, action=pi, detach_encoder=True)

            actor_pig_Q = torch.min(actor_pig_Q1, actor_pig_Q2)
            if self.loss_reduction == "mean":
                pig_actor_loss = ( -1 * actor_pig_Q
                ).mean()
                logger.log("train/pig_actor_loss", pig_actor_loss, step)

            actor_loss = actor_loss + pig_actor_loss


        logger.log("train/actor_target_entropy", self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )

        logger.log("train/actor_entropy", entropy.mean(), step)

        # optimize the actor
        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="actor"), 1)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        if self.loss_reduction == "mean":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            ).mean()
            logger.log("train/alpha_loss", alpha_loss, step)
        elif self.loss_reduction == "none":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            )
            logger.log("train/alpha_loss", alpha_loss.mean(), step)
        # breakpoint()
        # logger.log("train/alpha_value", self.get_alpha(batch.task_obs), step)
        self._compute_gradient(
            loss=alpha_loss,
            parameters=self.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        # torch.nn.utils.clip_grad_norm_(self.get_parameters(name="log_alpha"), 1)
        self.log_alpha_optimizer.step()

    def get_task_info(
        self, task_encoding: TensorType, component_name: str, env_index: TensorType
    ) -> TaskInfo:
        """Encode task encoding into task info.

        Args:
            task_encoding (TensorType): encoding of the task.
            component_name (str): name of the component.
            env_index (TensorType): index of the environment.

        Returns:
            TaskInfo: TaskInfo object.
        """
        if self.should_use_task_encoder:
            if component_name in self.multitask_cfg.task_encoder_cfg.losses_to_train:
                task_info = TaskInfo(
                    encoding=task_encoding, compute_grad=True, env_index=env_index
                )
            else:
                task_info = TaskInfo(
                    encoding=task_encoding.detach(),
                    compute_grad=False,
                    env_index=env_index,
                )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info

    def update_transition_reward_model(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the transition model and reward decoder.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the task encoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        self.task_encoder_optimizer.step()

    def update_decoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the decoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
        use_expert=False,
        expert_id=None,
        relabel_id=None,
        sync=False,
        use_pig=False
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """
        if sync:
            self.pig_critic.load_state_dict(self.critic.state_dict())
            self.pig_critic_target.load_state_dict(self.critic.state_dict())

        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if buffer_index_to_sample is None:
            batch = replay_buffer.sample()
        else:
            batch = replay_buffer.sample(buffer_index_to_sample)

        logger.log("train/batch_reward", batch.reward.mean(), step)
        if self.should_use_task_encoder:
            self.task_encoder_optimizer.zero_grad()
            task_encoding = self.get_task_encoding(
                env_index=batch.task_obs.squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            task_encoding = None  # type: ignore[assignment]

        task_info = self.get_task_info(
            task_encoding=task_encoding,
            component_name="critic",
            env_index=batch.task_obs,
        )
        self.update_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )
        if use_pig:
            self.update_pig_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if step % self.actor_update_freq == 0:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="actor",
                env_index=batch.task_obs,
            )
            self.update_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
                use_expert=use_expert,
                expert_id=expert_id,
                relabel_id=relabel_id,
                use_pig = use_pig
            )
        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.pig_critic.Q1, self.pig_critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.pig_critic.Q2, self.pig_critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if (
            "transition_model" in self._components
            and "reward_decoder" in self._components
        ):
            # some of the logic is a bit sketchy here. We will get to it soon.
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="transition_reward",
                env_index=batch.task_obs,
            )
            self.update_transition_reward_model(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if (
            "decoder" in self._components  # should_update_decoder
            and self.decoder is not None  # type: ignore[attr-defined]
            and step % self.decoder_update_freq == 0  # type: ignore[attr-defined]
        ):
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="decoder",
                env_index=batch.task_obs,
            )
            self.update_decoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        if self.should_use_task_encoder:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="task_encoder",
                env_index=batch.task_obs,
            )
            self.update_task_encoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        return batch.buffer_index

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]:
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            return list(self.actor.model.parameters())
        elif name in ["log_alpha", "alpha"]:
            return [self.log_alpha]
        elif name == "encoder":
            return list(self.critic.encoder.parameters())
        else:
            return list(self._components[name].parameters())