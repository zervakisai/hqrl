"""Actor heads of the recurrent mission policy: GRU features (..., h) -> unmasked action logits (..., A).

smdp_ppo.RecurrentActorCritic selects a head by name (model_config 'actor_head', default 'linear') and masks its logits.
The actor head decides during the mission: it runs at every policy decision of training, evaluation and deployment
(RecurrentActorCritic.policy), unlike the critic head, which only trains. Parameter counts below are for h = 64 and
A = 24 (obs-encoding/3, action-space/2); in general linear h*A + A, bottleneck 4h + 24 + 5A, fourier 4h + 4 + 17A,
quantum 4h + 20 + 5A (circuit weights 16), surrogate625 4h + 4 + 625*4 + 4 + 5A.

  linear                   Linear(h, A). The actor of every checkpoint written before actor heads existed (state_dict
                           keys actor.weight and actor.bias).                                                        1560
  bottleneck               Linear(h,4)-Tanh-Linear(4,4)-Tanh-Linear(4,A). Classical control with the quantum head's
                           4-dimensional bottleneck and nearly its parameter count.                                   400
  fourier                  angles = pi*tanh(Linear(h,4)); features sin and cos of angles*[1, 2] (4 wires x 2
                           frequencies x 2 = 16); Linear(16, A). A **separable Fourier-feature control**: it spans the
                           constant and the single-angle harmonics 1 and 2 only. It is NOT a dequantisation of the
                           quantum head and is not to be called one (use surrogate625 for that). Measured on the
                           shipped circuit (output/study/surrogate_625_v1/REPORT_EN.md): the features it can represent
                           carry 4.15% of the AC power of <Z_i> on average over weight draws, because the CNOT ring
                           puts the rest in products of harmonics of different angles.                               668
  quantum                  angles = pi*tanh(Linear(h,4)); 4 qubits, 2 layers, the circuit structure of
                           models.QuantumCritic: per layer and wire RY(angle), trainable RY and RZ, then a CNOT ring
                           (wire -> wire+1 mod 4); <Z> per wire; Linear(4, A). PennyLane default.qubit, shots=None,
                           diff_method='backprop' (analytic statevector). The last layer's 4 RZ angles are inert (their
                           gradient is exactly 0: RZ, the CNOT ring and the Z readout are all diagonal or permutations in
                           the computational basis), so 392 of the 396 parameters are effective, as for the critic
                           (audit F12).                                                                                396
  quantum_no_entanglement  the same circuit without the CNOT ring (the last-layer RZ angles are inert here too). A
                           **product-state control, not an entanglement ablation**: measured, every output collapses
                           to a + b*cos(2*theta_i) + c*sin(2*theta_i) of its own angle alone (3 of the 625 Fourier
                           coefficients, no cross-wire term and no first harmonic), so removing the ring removes the
                           entire cross-wire interaction structure rather than a marginal factor.                    396
  mlp                      Linear(h,64)-Tanh-Linear(64,64)-Tanh-Linear(64,A). C-strong of the GNSS study (statistical
                           protocol section 2): a practical nonlinear classical actor, deliberately far above the
                           matched count, so a quantum result against C-matched is read beside a strong classical one.
                                                                                                                  9880
  surrogate625             angles = pi*tanh(Linear(h,4)); the 5^4 = 625 explicit products of {1, cos t, sin t, cos 2t,
                           sin 2t} over the four angles; Linear(625, 4)-Linear(4, A). The **exact classical
                           dequantisation** of the quantum head: a linear readout on these features reproduces every
                           <Z_i> of the circuit to 1.8e-15 for any circuit weights, with no training (prediction P1,
                           output/study/surrogate_625_v1/REPORT_EN.md), so its hypothesis class contains the quantum
                           head's. Matched in function class, NOT in parameter count; the circuit reaches that class
                           through 16 weights on a low-dimensional manifold, this head with free coefficients.      2884

Initialisation order inside every head is the order above (projection, circuit weights 0.1*randn(2, 4, 2) or the
surrogate readout, readout), so a head consumes the global torch generator deterministically.

Finite-shot execution (quantum heads; simulated "run on quantum hardware" scenario, no device noise model)
    set_shots(shots, seed): shots None (default) is analytic execution, bit-identical to training. shots an integer in
    [1, MAX_SHOTS] requires seed an integer >= 0 (and shots None requires seed None) and resets the call counter to 0.
    MAX_SHOTS = 2**53 keeps every count and every sum of counts exact in float64. In finite-shot mode
    a forward call raises RuntimeError while torch grad mode is enabled and the module is in training mode, and otherwise
    runs the whole head (projection, circuit, readout) under torch.no_grad, so its logits carry no graph: it evaluates the
    analytic joint probabilities p of the 16 computational basis states
    (qml.probs over wires 0..3, wire 0 the most significant bit; clipped at 0 and renormalised per row), draws the counts
    of `shots` basis-state measurements (bitstrings) of every input row as counts = rng.multinomial(shots, p) with
    rng = numpy.random.default_rng([seed, call counter]) (p the (rows, 16) matrix, rows in input order, one generator per
    forward call), increments the counter, and estimates <Z_i> of each row as the mean of (1 - 2 bit_i) over its shots,
    counts @ (1 - 2 bits) / shots, on the grid 1 - 2k/shots. The counts of `shots` independent draws are multinomial, so
    this is the estimator of `shots` explicit bitstring samples, in O(16) memory and time per row whatever `shots` is. The
    readout then maps the estimates to logits. Outputs are therefore a deterministic function of the seed and
    of the sequence of forward calls since set_shots. The shot setting is an execution setting: it is not part of
    model_config and is never saved in a checkpoint, and the head description records the training device
    (training_shots None, analytic); smdp_ppo.RecurrentActorCritic.describe()['actor_head']['execution_shots'] records the
    current execution setting.
    Repeated episodes: set_shots restarts the stream at call 0, so two episodes that each call set_shots with the same
    seed share their shot noise (common random numbers). Independent shot noise per repeat needs a different seed per
    repeat (run_mission.build_rl_policy).

Circuit accounting (quantum heads)
    calls: circuit executions (one per forward call, analytic or probability); rows: angle rows sent to the circuit;
    snapshot() = (calls, rows), the interface of train_smdp_ppo.CircuitCounter; shot_calls: finite-shot forward calls since
    the last set_shots.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

ACTOR_HEADS = ('linear', 'bottleneck', 'fourier', 'quantum', 'quantum_no_entanglement', 'surrogate625', 'mlp')
#: C-strong (statistical protocol section 2): a practical nonlinear actor, not merely a larger linear layer.
MLP_WIDTH = 64
QUANTUM_ACTOR_HEADS = ('quantum', 'quantum_no_entanglement')
WIRES, LAYERS = 4, 2
SURROGATE_BASIS = 2 * LAYERS + 1               # {1, cos t, sin t, cos 2t, sin 2t}: the spectrum of LAYERS data re-uploads
SURROGATE_FEATURES = SURROGATE_BASIS ** WIRES  # 625; both follow WIRES and LAYERS, so surrogate625 cannot silently go stale
STATES = 2 ** WIRES
MAX_SHOTS = 2 ** 53
BASIS_BITS = np.asarray([[(state >> (WIRES - 1 - wire)) & 1 for wire in range(WIRES)] for state in range(STATES)], dtype=np.float64)
BASIS_Z = 1. - 2. * BASIS_BITS  # (16, 4): the eigenvalue of Z_i on each basis state


class FourierActorHead(nn.Module):
    """Separable Fourier-feature actor head (module docstring, 'fourier')."""

    def __init__(self, hidden, action_dim):
        super().__init__()
        self.projection = nn.Linear(hidden, WIRES)
        self.register_buffer('frequencies', torch.tensor([1., 2.]))
        self.readout = nn.Linear(WIRES * 4, action_dim)

    def forward(self, features):
        phases = (math.pi * torch.tanh(self.projection(features))).unsqueeze(-1) * self.frequencies
        return self.readout(torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(-2))


class SurrogateActorHead(nn.Module):
    """Exact classical surrogate of the quantum head (module docstring, 'surrogate625')."""

    def __init__(self, hidden, action_dim):
        super().__init__()
        self.projection = nn.Linear(hidden, WIRES)
        self.surrogate = nn.Linear(SURROGATE_FEATURES, WIRES)   # replaces the circuit: 625 features -> 4 values
        self.readout = nn.Linear(WIRES, action_dim)

    @staticmethod
    def features(angles):
        """(rows, WIRES) angles -> (rows, SURROGATE_FEATURES) products over the wires of (1, cos t, sin t, cos 2t, sin 2t).

        Harmonics follow LAYERS and wires follow WIRES, so the basis cannot go stale if the circuit changes shape; wire 0
        is the slowest index, so feature (a, b, c, d) sits at a*125 + b*25 + c*5 + d.
        """
        harmonics = [torch.ones_like(angles)]
        for order in range(1, LAYERS + 1):
            harmonics += [(order * angles).cos(), (order * angles).sin()]
        basis = torch.stack(harmonics, dim=-1)                       # (rows, WIRES, 2*LAYERS+1)
        products = basis[:, 0]
        for wire in range(1, WIRES):
            products = (products.unsqueeze(-1) * basis[:, wire].unsqueeze(-2)).reshape(angles.shape[0], -1)
        return products

    def forward(self, features):
        shape = features.shape[:-1]
        angles = math.pi * torch.tanh(self.projection(features.reshape(-1, features.shape[-1])))
        return self.readout(self.surrogate(self.features(angles))).reshape(*shape, -1)


class QuantumActorHead(nn.Module):
    """Four-qubit, two-layer variational actor head with optional finite-shot execution (module docstring)."""

    def __init__(self, hidden, action_dim, entangle=True):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError('Install pennylane to use a quantum actor head') from exc
        self.entangle = entangle = bool(entangle)
        self.projection = nn.Linear(hidden, WIRES)
        self.weights = nn.Parameter(0.1 * torch.randn(LAYERS, WIRES, 2))
        self.readout = nn.Linear(WIRES, action_dim)
        self.shots, self.seed, self.shot_calls = None, None, 0
        self.calls, self.rows = 0, 0
        device = qml.device('default.qubit', wires=WIRES, shots=None)

        def layers(angles, weights):
            for layer in range(LAYERS):
                for wire in range(WIRES):
                    qml.RY(angles[..., wire], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                if entangle:
                    for wire in range(WIRES):
                        qml.CNOT(wires=[wire, (wire + 1) % WIRES])

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(angles, weights):
            layers(angles, weights)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(WIRES)]

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def probability_circuit(angles, weights):
            layers(angles, weights)
            return qml.probs(wires=list(range(WIRES)))

        self.circuit, self.probability_circuit = circuit, probability_circuit

    def set_shots(self, shots: Optional[int], seed: Optional[int]):
        """Analytic (shots None, seed None) or finite-shot execution with a seed; resets the finite-shot call counter."""
        if shots is None:
            if seed is not None:
                raise ValueError('seed: analytic execution (shots None) takes no seed')
        else:
            if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or not 1 <= shots <= MAX_SHOTS:
                raise ValueError(f'shots: expected None or an integer in [1, {MAX_SHOTS}]')
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError('seed: finite-shot execution requires an integer seed >= 0')
            shots, seed = int(shots), int(seed)
        self.shots, self.seed, self.shot_calls = shots, seed, 0

    def snapshot(self):
        return self.calls, self.rows

    def _sampled_expectations(self, angles):
        probabilities = self.probability_circuit(angles, self.weights)
        p = probabilities.detach().cpu().to(torch.float64).numpy().reshape(-1, STATES).clip(0., None)
        rng = np.random.default_rng([self.seed, self.shot_calls])
        self.shot_calls += 1
        counts = rng.multinomial(self.shots, p / p.sum(axis=1, keepdims=True))
        return torch.from_numpy(counts @ BASIS_Z / self.shots)

    def _logits(self, features, shape):
        angles = math.pi * torch.tanh(self.projection(features.reshape(-1, features.shape[-1])))
        if self.shots is None:
            expectations = torch.stack(self.circuit(angles, self.weights), dim=-1)
        else:
            expectations = self._sampled_expectations(angles)
        self.calls += 1
        self.rows += int(angles.shape[0])
        expectations = expectations.to(dtype=features.dtype, device=features.device)
        return self.readout(expectations).reshape(*shape, -1)

    def forward(self, features):
        shape = features.shape[:-1]
        if self.shots is None:
            return self._logits(features, shape)
        if self.training and torch.is_grad_enabled():
            raise RuntimeError('finite-shot execution has no gradient: call eval() or run under torch.no_grad()')
        with torch.no_grad():  # the whole head, readout included: the output carries no graph
            return self._logits(features, shape)


def build_actor_head(name, hidden, action_dim):
    """nn.Module mapping (..., hidden) GRU features to (..., action_dim) logits (module docstring)."""
    if name == 'linear':
        return nn.Linear(hidden, action_dim)
    if name == 'bottleneck':
        return nn.Sequential(nn.Linear(hidden, WIRES), nn.Tanh(), nn.Linear(WIRES, WIRES), nn.Tanh(), nn.Linear(WIRES, action_dim))
    if name == 'fourier':
        return FourierActorHead(hidden, action_dim)
    if name == 'surrogate625':
        return SurrogateActorHead(hidden, action_dim)
    if name in QUANTUM_ACTOR_HEADS:
        return QuantumActorHead(hidden, action_dim, entangle=name == 'quantum')
    if name == 'mlp':
        return nn.Sequential(nn.Linear(hidden, MLP_WIDTH), nn.Tanh(), nn.Linear(MLP_WIDTH, MLP_WIDTH), nn.Tanh(),
                             nn.Linear(MLP_WIDTH, action_dim))
    raise ValueError(f'actor_head: expected one of {ACTOR_HEADS}, got {name!r}')


def actor_head_description(name):
    """Architecture record of an actor head for model descriptions and run records."""
    if name == 'linear':
        return dict(kind='linear', layers='Linear(h,A)')
    if name == 'bottleneck':
        return dict(kind='mlp', layers='Linear(h,4)-Tanh-Linear(4,4)-Tanh-Linear(4,A)',
                    role='classical control with the quantum head bottleneck')
    if name == 'mlp':
        return dict(kind='mlp', layers=f'Linear(h,{MLP_WIDTH})-Tanh-Linear({MLP_WIDTH},{MLP_WIDTH})-Tanh-Linear({MLP_WIDTH},A)',
                    role='C-strong: a practical nonlinear classical actor, far above the matched parameter count')
    if name == 'fourier':
        return dict(kind='fourier', implementation='aerial_qrl.training.actor_heads.FourierActorHead',
                    encoding='angles = pi*tanh(Linear(h,4)); sin and cos of angles*[1,2]', readout='Linear(16,A)',
                    role='separable Fourier-feature control: constant and single-angle harmonics only; NOT a dequantisation '
                         '(measured share of the circuit AC power 4.15%, output/study/surrogate_625_v1)')
    if name == 'surrogate625':
        return dict(kind='fourier_surrogate', implementation='aerial_qrl.training.actor_heads.SurrogateActorHead',
                    encoding='angles = pi*tanh(Linear(h,4)); the 5^4 products of (1, cos t, sin t, cos 2t, sin 2t) over the 4 angles',
                    features=SURROGATE_FEATURES, readout='Linear(625,4)-Linear(4,A)',
                    role='exact classical dequantisation of the quantum head: its hypothesis class contains the circuit\'s '
                         '(P1 verified to 1.8e-15, output/study/surrogate_625_v1); matched in function class, not in parameters')
    if name in QUANTUM_ACTOR_HEADS:
        if name == 'quantum_no_entanglement':
            role = ('product-state control: without the CNOT ring every output is a + b*cos(2*theta_i) + c*sin(2*theta_i) of '
                    'its own angle (3 of 625 Fourier coefficients, measured), so this removes the whole cross-wire structure')
        else:
            role = 'the hybrid arm; its hypothesis class is reproduced exactly by the surrogate625 head (P1)'
        return dict(kind='variational_circuit', implementation='aerial_qrl.training.actor_heads.QuantumActorHead', role=role, qubits=WIRES,
                    layers=LAYERS, encoding='angles = pi*tanh(Linear(h,4)); every layer RY(angle), trainable RY and RZ per wire',
                    entanglement='CNOT ring' if name == 'quantum' else None, measurement='<Z> per wire', readout='Linear(4,A)',
                    backend='pennylane default.qubit', training_shots=None, diff_method='backprop', inert_parameters='last-layer RZ (4)',
                    finite_shot_execution='set_shots(shots, seed): qml.probs, multinomial counts of shots basis-state draws per '
                                          'row with numpy default_rng([seed, call]); no gradient; the current setting is '
                                          'execution_shots (None: analytic)')
    raise ValueError(f'actor_head: expected one of {ACTOR_HEADS}, got {name!r}')


__all__ = ['ACTOR_HEADS', 'MLP_WIDTH', 'MAX_SHOTS', 'QUANTUM_ACTOR_HEADS', 'SURROGATE_FEATURES', 'FourierActorHead', 'QuantumActorHead',
           'SurrogateActorHead', 'actor_head_description', 'build_actor_head']
