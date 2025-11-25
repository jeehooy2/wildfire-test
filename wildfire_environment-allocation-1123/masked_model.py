"""
Masked Categorical Model for RLlib with Action Masking Support

This model applies action masking to prevent the agent from selecting invalid actions.
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class MaskedCategoricalTorchModel(TorchModelV2, nn.Module):
    """Custom PyTorch model that supports action masking for categorical actions.

    This model takes observations in Dict format with:
    - "obs": The actual observation vector
    - "action_mask": Binary mask indicating valid actions (1) and invalid actions (0)

    The model applies the mask by adding large negative values to logits of invalid actions,
    effectively preventing them from being selected.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the masked categorical model.

        Parameters
        ----------
        obs_space : gym.Space
            Observation space (should be Dict with "obs" and "action_mask")
        action_space : gym.Space
            Action space (should be Discrete)
        num_outputs : int
            Number of output units (should match action_space.n)
        model_config : dict
            Model configuration dictionary
        name : str
            Model name
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.n_actions = action_space.n

        # Get observation dimension from the "obs" key in Dict space
        obs_dim = int(np.prod(obs_space["obs"].shape))

        # Neural network layers
        hidden_size = model_config.get("fcnet_hiddens", [256, 256])

        layers = []
        prev_size = obs_dim
        for h_size in hidden_size:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU()
            ])
            prev_size = h_size

        self.encoder = nn.Sequential(*layers)

        # Policy head (actor)
        self.pi = nn.Linear(prev_size, self.n_actions)

        # Value head (critic)
        self.v = nn.Linear(prev_size, 1)

        # Store value function output
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing observations
        state : list
            RNN state (not used in this model)
        seq_lens : Tensor
            Sequence lengths for RNN (not used in this model)

        Returns
        -------
        tuple
            (masked_logits, state) where masked_logits has invalid actions masked out
        """
        # Extract observation and action mask from input
        obs = input_dict["obs"]["obs"].float()
        mask = input_dict["obs"]["action_mask"].float()

        # Flatten observation if needed
        if len(obs.shape) > 2:
            obs = obs.view(obs.size(0), -1)

        # Encode observation
        x = self.encoder(obs)

        # Compute logits
        logits = self.pi(x)

        # Apply action mask: convert mask to log probabilities
        # mask is 1 for valid actions, 0 for invalid
        # log(0) = -inf, so we clamp to a large negative value
        inf_mask = torch.clamp(torch.log(mask), min=-1e9)
        masked_logits = logits + inf_mask

        # Compute value function
        self._value = self.v(x).squeeze(1)

        return masked_logits, state

    def value_function(self):
        """Return the value function output from the last forward pass.

        Returns
        -------
        Tensor
            Value function estimates
        """
        return self._value
