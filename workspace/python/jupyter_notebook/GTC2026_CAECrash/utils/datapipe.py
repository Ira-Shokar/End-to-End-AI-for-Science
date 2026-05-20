# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
from typing import Any, Callable, Optional
from physicsnemo.datapipes.gnn.utils import load_json, save_json

# Handle missing physicsnemo.utils.logging module
try:
    from physicsnemo.utils.logging import PythonLogger
except (ImportError, ModuleNotFoundError):
    class PythonLogger:
        def __init__(self, name='root'):
            self.name = name
        def info(self, msg):
            print(f"[INFO] {msg}")
        def warning(self, msg):
            print(f"[WARNING] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")
        def debug(self, msg):
            print(f"[DEBUG] {msg}")

STATS_DIRNAME = "stats"
NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EDGE_STATS_FILE = "edge_stats.json"
EPS = 1e-8 

class SimSample:
    """Unified representation for Simulation data."""
    def __init__(
        self,
        node_features: dict[str, torch.Tensor],
        node_target: torch.Tensor,
        graph = None,
        global_features: Optional[dict[str, torch.Tensor]] = None,
    ):
        self.node_features = node_features
        self.node_target = node_target
        self.graph = graph  
        self.global_features = global_features

    def to(self, device: torch.device):
        for k, v in self.node_features.items():
            self.node_features[k] = v.to(device)
        self.node_target = self.node_target.to(device)
        if self.graph is not None:
            self.graph = self.graph.to(device)
        if self.global_features is not None:
            self.global_features = {k: v.to(device) for k, v in self.global_features.items()}
        return self

class CrashBaseDataset:
    def __init__(
        self,
        name: str = "dataset",
        reader: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        global_features_filepath: Optional[str] = None,
        global_features: Optional[list[str]] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 400,
        features: Optional[list[str]] = None,
        logger=None,
        dt: float = 5e-3,
        physics_stats: Optional[dict] = None, # ADDED: Allow passing global maximums
    ):
        super().__init__()
        self.name = name
        self.data_dir = data_dir or "."
        self.global_features_filepath = global_features_filepath
        self.global_features_keys = global_features
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.features = features or []
        self.logger = logger or PythonLogger()
        self.dt = dt
        self.physics_stats = physics_stats # ADDED: Store stats

        self._stats_dir = STATS_DIRNAME
        os.makedirs(STATS_DIRNAME, exist_ok=True)

        if reader is None:
            raise ValueError("Data reader function is not specified.")
        
        # Load raw records
        self.srcs, self.dsts, point_data, global_features_raw = reader(
            data_dir=self.data_dir,
            num_samples=num_samples,
            split=split,
            global_features_filepath=self.global_features_filepath,
            logger=self.logger,
        )
        
        # Set length based on actual records loaded
        self.length = len(point_data)
        self.num_samples = self.length

        # Persistence: Store raw records for build_xy access
        self.raw_records = point_data

        if global_features_raw is not None and self.global_features_keys is not None:
            self.global_features = []
            for i, gf in enumerate(global_features_raw):
                self.global_features.append({k: gf[k] for k in self.global_features_keys})
        else:
            self.global_features = None

        self.mesh_pos_seq: list[torch.Tensor] = []  
        self.node_features_data: list[torch.Tensor] = []  

        for rec in point_data:
            coords_np = rec["coords"][:num_steps]
            self.mesh_pos_seq.append(torch.as_tensor(coords_np, dtype=torch.float32))

            parts = []
            for k in self.features:
                arr = rec[k]
                if arr.ndim == 1: arr = arr[:, None]
                # If temporal, take first slice for static feature input
                if arr.ndim == 2 and arr.shape[1] > 1 and arr.shape[0] == coords_np.shape[1]:
                    arr = arr[:, 0:1]
                parts.append(arr)

            feats_np = np.concatenate(parts, axis=-1) if len(parts) > 0 else np.zeros((coords_np.shape[1], 0), dtype=np.float32)
            self.node_features_data.append(torch.as_tensor(feats_np, dtype=torch.float32))

        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train":
            self.node_stats = self._compute_autoreg_node_stats()
            self.feature_stats = self._compute_feature_stats()
            save_json(self.node_stats, node_stats_path)
            save_json(self.feature_stats, feat_stats_path)
        else:
            self.node_stats = load_json(node_stats_path)
            self.feature_stats = load_json(feat_stats_path)

        for i in range(len(self.mesh_pos_seq)):
            self.mesh_pos_seq[i] = self._normalize_node_tensor(
                self.mesh_pos_seq[i], 
                self.node_stats["pos_mean"], 
                self.node_stats["pos_std"]
            )

    def __len__(self):
        return self.length

    def _xy_shapes(self, idx: int, time_history: int = 1) -> tuple[int, int]:
        T, N, _ = self.mesh_pos_seq[idx].shape
        Dout = (T - time_history) * 5 
        Din = 5 * time_history # Updated to 10 for time_history=2 (6 pos + 4 phys)
        return Din, Dout

    def build_xy(self, idx: int, time_history: int = 1):
        assert 0 <= idx < self.num_samples, f"Index {idx} out of range"
        pos_seq = self.mesh_pos_seq[idx]  # [T, N, 3]
        feats = self.node_features_data[idx] 
        rec = self.raw_records[idx] 
        
        T, N, _ = pos_seq.shape

        # 1. Prepare secondary physical fields (Strain/Stress)
        strain = torch.as_tensor(rec["plastic_strain"], dtype=torch.float32)
        stress = torch.as_tensor(rec["von_mises_stress"], dtype=torch.float32)
        
        # Ensure temporal dimension is first: [T, N]
        if strain.shape == (N, T) or (strain.ndim == 2 and strain.shape[0] == N):
            strain, stress = strain.T, stress.T
            
        strain = strain[:T].reshape(T, N, 1)
        stress = stress[:T].reshape(T, N, 1)

        # --- NEW: GLOBAL MAX NORMALIZATION ---
        # Divide by global max to map [0, Max] -> [0.0, 1.0]
        if self.physics_stats is not None:
            strain = strain / (self.physics_stats["plastic_strain_max"] + EPS)
            stress = stress / (self.physics_stats["von_mises_stress_max"] + EPS)
        # -------------------------------------

        physics_seq = torch.cat([strain, stress], dim=-1) # [T, N, 2]

        # 2. Slice History for Input X
        hist_pos = pos_seq[:time_history]
        hist_physics = physics_seq[:time_history]

        # 3. Define Target Y 
        # Because we concatenated pos_seq and the normalized physics_seq, 
        # the target is automatically completely normalized!
        target_traj = torch.cat([pos_seq, physics_seq], dim=-1) # [T, N, 5]
        y = target_traj[time_history:].transpose(0, 1).flatten(start_dim=1) 

        x = {
            "coords": hist_pos,       # [H, N, 3]
            "features": hist_physics,  # [H, N, 2]
            "static_feats": feats      # [N, F]
        }

        return x, y

    def _compute_autoreg_node_stats(self):
        dt = self.dt
        pos_mean = torch.zeros(3)
        pos_meansqr = torch.zeros(3)
        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]
            pos_mean += torch.mean(pos, dim=(0, 1)) / self.num_samples
            pos_meansqr += torch.mean(pos * pos, dim=(0, 1)) / self.num_samples
        pos_std = torch.sqrt(torch.clamp(pos_meansqr - pos_mean * pos_mean, min=0.0) + EPS)

        return {
            "pos_mean": pos_mean, 
            "pos_std": pos_std, 
            "norm_vel_mean": torch.zeros(3), 
            "norm_vel_std": torch.ones(3),
            "norm_acc_mean": torch.zeros(3), 
            "norm_acc_std": torch.ones(3)
        }

    def _compute_feature_stats(self):
        return {"feature_mean": torch.zeros(0), "feature_std": torch.ones(0)}

    @staticmethod
    def _normalize_node_tensor(invar, mu, std):
        mu = torch.as_tensor(mu).view(1, 1, -1)
        std = torch.as_tensor(std).view(1, 1, -1)
        return (invar - mu) / (std + EPS)


class CrashPointCloudDataset(CrashBaseDataset):
    def __init__(self, *args, **kwargs):
        # ADDED: Extract physics_stats and pass it down
        physics_stats = kwargs.pop("physics_stats", None)
        super().__init__(*args, physics_stats=physics_stats, **kwargs)
        self.edge_stats = {}

    def __getitem__(self, idx: int):
        x, y = self.build_xy(idx)
        if self.global_features is not None:
            gf = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.global_features[idx].items()}
        else:
            gf = None
        return SimSample(node_features=x, node_target=y, global_features=gf)

def simsample_collate(batch: list[SimSample]) -> list[SimSample]:
    return batch