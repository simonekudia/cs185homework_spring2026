"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        layers= []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim,  action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[1], action_dim * chunk_size)
        # )
        self.loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_action_chunk = self.net(state).view(-1, self.chunk_size, self.action_dim)
        return self.loss(predicted_action_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.net(state).view(-1, self.chunk_size, self.action_dim)
        


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers= []
        in_dim = state_dim + (action_dim * chunk_size) + 1 #state_dim + action_chunk_noise + tao
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim,  action_dim * chunk_size))
        self.net = nn.Sequential(*layers)
        self.loss = nn.MSELoss()


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        #sample tao shape (batch size, 1, 1), similar to action chunk
        batch_size = state.shape[0]
        tao = torch.rand(batch_size, 1, 1, device=state.device)
        #sample action chunk noise
        action_chunk_noise = torch.randn_like(action_chunk)
        #interpolate
        tao_sample_point = tao * action_chunk + (1-tao) * action_chunk_noise
        #vtheta takes in state, interpolation point, and tao
        net_in = torch.cat([state, tao_sample_point.view(batch_size, -1), tao.squeeze(-1)], dim=-1)
        sample_velocity = self.net(net_in)
        actual_velocity = (action_chunk - action_chunk_noise).view(batch_size,self.action_dim*self.chunk_size)
        return self.loss(sample_velocity, actual_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        #tau = torch.rand()
        action_chunk = torch.randn(batch_size, self.chunk_size * self.action_dim, device=state.device)

        delta_tau = 1.0 / num_steps
        for i in range(num_steps):
            tau = i / num_steps
            t = torch.full((batch_size, 1), tau, device=state.device)

            net_in= torch.cat([state, action_chunk, t], dim=-1)
            velocity = self.net(net_in)

            action_chunk = action_chunk + delta_tau * velocity

        return action_chunk.view(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
