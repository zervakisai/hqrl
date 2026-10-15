"""Classical recurrent policy with interchangeable training-time value heads.

The quantum option simulates a small variational circuit. It is not a claim of
quantum speedup, and the actor never evaluates that circuit at deployment.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class FourierCritic(nn.Module):
    """Learned four-dimensional projection and explicit Fourier features."""

    def __init__(self, hidden: int, bottleneck: int = 4):
        super().__init__()
        self.projection = nn.Linear(hidden, bottleneck)
        self.register_buffer("frequencies", torch.tensor([1., 2., 3.]))
        self.readout = nn.Sequential(nn.Linear(bottleneck * 6, hidden),
                                     nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features):
        angles = math.pi * torch.tanh(self.projection(features))
        phases = angles.unsqueeze(-1) * self.frequencies
        encoded = torch.cat((phases.sin(), phases.cos()), dim=-1)
        return self.readout(encoded.flatten(-2))


class QuantumCritic(nn.Module):
    """Four-qubit, two-layer, analytic-statevector quantum value head.

Finite-shot/noisy training requires a separate experiment; this implementation
only supplies deterministic differentiable simulation with backpropagation.
"""

    def __init__(self, hidden: int, entangle: bool = True):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install pennylane to use the quantum critic") from exc
        self.projection = nn.Linear(hidden, 4)
        self.weights = nn.Parameter(0.1 * torch.randn(2, 4, 2))
        self.readout = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        device = qml.device("default.qubit", wires=4, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(angles, weights):
            for layer in range(2):
                for wire in range(4):
                    qml.RY(angles[..., wire], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                if entangle:
                    for wire in range(4):
                        qml.CNOT(wires=[wire, (wire + 1) % 4])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

        self.circuit = circuit

    def forward(self, features):
        shape = features.shape[:-1]
        angles = math.pi * torch.tanh(self.projection(features.reshape(-1, features.shape[-1])))
        expectations = torch.stack(self.circuit(angles, self.weights), dim=-1)
        expectations = expectations.to(dtype=features.dtype, device=features.device)
        return self.readout(expectations).reshape(*shape, 1)


class ActorCritic(nn.Module):
    """Input B,T,D; output action logits B,T,A, values B,T, GRU state 1,B,H."""

    def __init__(self, obs_dim: int, action_dim: int, critic: str = "mlp", hidden: int = 64,
                 report_constraints: bool = False):
        super().__init__()
        if min(obs_dim, action_dim, hidden) <= 0:
            raise ValueError("Dimensions must be positive")
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, critic=critic, hidden=hidden,
                           report_constraints=report_constraints)
        self.report_constraints = report_constraints
        if report_constraints and (obs_dim,action_dim)!=(16,9):
            raise ValueError('Report constraints require the urban 16D/9-action interface')
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Linear(hidden, action_dim)
        if critic == "mlp":
            self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        elif critic == "fourier":
            self.critic = FourierCritic(hidden)
        elif critic in {"quantum", "hybrid", "qrl", "quantum_no_entanglement"}:
            self.critic = QuantumCritic(hidden, entangle=critic != "quantum_no_entanglement")
        elif critic == "small":
            self.critic = nn.Sequential(nn.Linear(hidden, 4), nn.Tanh(),
                                        nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            raise ValueError(f"Unknown critic: {critic}")

    def initial_state(self, batch: int):
        parameter = next(self.parameters())
        return torch.zeros(1, batch, self.hidden, dtype=parameter.dtype, device=parameter.device)

    def policy(self, obs, state=None):
        """Deployment path: classical actor only, no critic/circuit evaluation."""
        if obs.ndim != 3:
            raise ValueError("Expected observations with shape (batch, time, obs_dim)")
        features, state = self.gru(self.encoder(obs), state)
        return self._constrain(self.actor(features),obs), state

    def _constrain(self,logits,obs):
        if not self.report_constraints:
            return logits
        allowed=torch.ones_like(logits,dtype=torch.bool)
        # Local observation contract only: no evaluator correctness or target truth.
        allowed[...,7]=(obs[...,9]>0.5)&(obs[...,14]<0.5)
        extent=obs.new_tensor([60.,60.,25.])
        allowed[...,8]=torch.linalg.vector_norm(obs[...,:3]*extent,dim=-1)<=2.
        return logits.masked_fill(~allowed,-1e9)

    def forward(self, obs, state=None):
        if obs.ndim != 3:
            raise ValueError("Expected observations with shape (batch, time, obs_dim)")
        features, state = self.gru(self.encoder(obs), state)
        return self._constrain(self.actor(features),obs), self.critic(features).squeeze(-1), state
