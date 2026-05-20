# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.models.transolver import Transolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.experimental.models.geotransolver import GeoTransolver

from datapipe import SimSample

EPS = 1e-8


class GeoTransolverRolloutTraining(GeoTransolver):
    """
    GeoTransolver model with autoregressive rollout training.

    Updated to support multi-output features (e.g., Position + Plastic Strain + Stress).
    The model predicts coordinate deltas for the first 3 channels and absolute
    values for subsequent physical properties.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, Fo] rollout of predicted positions and physical fields
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N, 3] or [H, N, 3] (time history)
        features = inputs.get("features", coords.new_zeros((*coords.shape[:-1], 0)))
        
        # Prepare global features
        global_features = torch.stack([sample.global_features[k] for k in sample.global_features.keys()], dim=0)
        
        # Number of nodes: coords is [N, 3] or [H, N, 3]
        N = coords.shape[-2] if coords.dim() == 3 else coords.size(0)

        def step_fn(fx, embedding, global_embedding):
            return super(GeoTransolverRolloutTraining, self).forward(
                local_embedding=fx, 
                geometry=embedding, 
                local_positions=embedding, 
                global_embedding=global_embedding
            )

        # Build input: typically just [N, 3] if features list is empty
        fx_t = torch.cat([coords, features], dim=-1)

        # Forward pass: outf is [N, rollout_steps * out_channels]
        outf = step_fn(
            fx_t, 
            coords, 
            global_features.unsqueeze(0).unsqueeze(0)
        ).squeeze(0)

        # Determine output dimensionality (e.g., 5 channels: x, y, z, strain, stress)
        out_channels = outf.shape[-1] // self.rollout_steps
        
        # Reshape to [Nodes, Time, Channels]
        out_reshaped = outf.reshape(N, self.rollout_steps, out_channels)

        # Initial positions for delta decoding: [N, 3] (use last timestep if coords is [H, N, 3])
        coords_0 = coords[-1] if coords.dim() == 3 else coords

        # --- Channel Splitting ---
        # 1. First 3 channels are position deltas: Add to initial coordinates
        pos_out = coords_0.unsqueeze(1) + out_reshaped[:, :, :3]
        
        # 2. Remaining channels are physics (stress/strain): Absolute values
        if out_channels > 3:
            physics_out = out_reshaped[:, :, 3:]
            outputs = torch.cat([pos_out, physics_out], dim=-1)
        else:
            outputs = pos_out

        # Return as [T, N, Fo]
        return outputs.transpose(0, 1).contiguous()


class TransolverAutoregressiveRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        inputs = sample.node_features
        coords = inputs["coords"]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        N = coords.size(0)
        device = coords.device

        y_t1 = coords
        y_t0 = y_t1 - self.initial_vel * self.dt

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            time_t = 0.0 if self.rollout_steps <= 1 else t / (self.rollout_steps - 1)
            time_t = torch.tensor([time_t], device=device, dtype=torch.float32)

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            fx_t = torch.cat([vel_norm, features, time_t.expand(N, 1)], dim=-1)

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            acc = outf * data_stats["node"]["norm_acc_std"] + data_stats["node"]["norm_acc_mean"]
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)


class TransolverTimeConditionalRollout(Transolver):
    """
    Transolver model with time-conditional rollout.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        inputs = sample.node_features
        x = inputs["coords"]
        features = inputs.get("features", x.new_zeros((x.size(0), 0)))

        outputs: list[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time in time_seq:
            fx_t = features

            def step_fn(fx, embedding, time_t):
                return super(TransolverTimeConditionalRollout, self).forward(
                    fx=fx, embedding=embedding, time=time_t
                )

            if self.training:
                outf = ckpt(step_fn, fx_t.unsqueeze(0), x.unsqueeze(0), time.unsqueeze(0), use_reentrant=False).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), x.unsqueeze(0), time.unsqueeze(0)).squeeze(0)

            y_t2 = x + outf
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0)


class MeshGraphNetAutoregressiveRolloutTraining(MeshGraphNet):
    """MeshGraphNet with autoregressive rollout training."""

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        initial_vel = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)
        
        if isinstance(initial_vel, torch.Tensor):
            self.register_buffer("initial_vel", initial_vel)
        else:
            self.register_buffer("initial_vel", torch.as_tensor(initial_vel, dtype=torch.float32))

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        inputs = sample.node_features
        coords = inputs["coords"]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        N = coords.size(0)
        device = coords.device
        
        y_t1 = coords
        y_t0 = y_t1 - self.initial_vel * self.dt

        outputs: list[torch.Tensor] = []
        for _ in range(self.rollout_steps):
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (data_stats["node"]["norm_vel_std"] + EPS)
            fx_t = torch.cat([y_t1, vel_norm, features], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetAutoregressiveRolloutTraining, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            outf = ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False) if self.training else step_fn(fx_t, edge_features, graph)
            acc = outf * data_stats["node"]["norm_acc_std"] + data_stats["node"]["norm_acc_mean"]
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)