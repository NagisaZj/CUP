# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Datastructure to wrap environment observation, task observation and other task-related information."""
from dataclasses import dataclass
from typing import Optional

from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import TensorType


@dataclass
class MTObs:
    """Class to wrap environment observation, task observation and other task-related information."""

    __slots__ = ["env_obs", "task_obs", "task_info"]
    env_obs: TensorType
    task_obs: Optional[TensorType]
    task_info: Optional[TaskInfo]

@dataclass
class InfoMTObs:
    """Class to wrap environment observation, task observation and other task-related information."""

    __slots__ = ["env_obs", "task_obs", "task_info",'info','t']
    env_obs: TensorType
    task_obs: Optional[TensorType]
    task_info: Optional[TaskInfo]
    info: Optional
    t: Optional
