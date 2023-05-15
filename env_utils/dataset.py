# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, states_grad=None, states_joint=None, early_term=None, shuffle = False, drop_last = False):
        self.obs = obs.view(-1, obs.shape[-1])
        if states_joint is not None:
            self.states = states_joint.view(-1, states_joint.shape[-1])
        else:
            self.states = states_joint
        if states_grad is not None:
            self.states_grad = states_grad.view(-1, states_grad.shape[-1])
        else:
            self.states_grad = states_grad
        if early_term is not None:
            self.early_term = early_term.view(-1)
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]
        if self.states is not None:
            self.states = self.states[index]
            self.states_grad = self.states_grad[index]
            self.early_term = self.early_term[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        if self.states is not None:
            return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx], 'states': self.states[start_idx:end_idx], 'states_grad': self.states_grad[start_idx:end_idx], 'early_term': self.early_term[start_idx:end_idx]}
        else:
            return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}
