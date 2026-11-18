"""Q-learning trainer, target network and validation for GSP-RGNN V2."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from config import Config
from environment import MazeInstance
from model import RelationalQNetwork, build_rgnn
from relational_graph import (
    RelationalGraphBatch,
    RelationalGraphBuilder,
    batch_relational_graphs,
)
from search import QGuidedWAStar, SearchResult, TransitionExperience


CATEGORY_UNSOLVED = "unsolved"
CATEGORY_SOLVED = "solved"
CATEGORY_SATISFICED = "satisficed"
INSTANCE_CATEGORIES = (CATEGORY_UNSOLVED, CATEGORY_SOLVED, CATEGORY_SATISFICED)


@dataclass
class UpdateStats:
    loss: float
    q_mean: float
    q_min: float
    q_max: float
    gradient_norm: float
    target_network_updated: bool
    gradient_updates: int
    num_action_objects: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "q_mean": self.q_mean,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "gradient_norm": self.gradient_norm,
            "target_network_updated": self.target_network_updated,
            "gradient_updates": self.gradient_updates,
            "num_action_objects": self.num_action_objects,
        }


@dataclass
class ValidationMetrics:
    num_instances: int
    num_solved: int
    success_rate: float  # percent, 0..100
    mean_pcr_successful: Optional[float]
    mean_path_cost_successful: Optional[float]
    mean_unique_state_expansions: float
    mean_reexpansions: float
    mean_transition_evaluations: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_instances": self.num_instances,
            "num_solved": self.num_solved,
            "success_rate": self.success_rate,
            "mean_pcr_successful": self.mean_pcr_successful,
            "mean_path_cost_successful": self.mean_path_cost_successful,
            "mean_unique_state_expansions": self.mean_unique_state_expansions,
            "mean_reexpansions": self.mean_reexpansions,
            "mean_transition_evaluations": self.mean_transition_evaluations,
        }

    def selection_key(self) -> Tuple[float, float, float]:
        pcr = self.mean_pcr_successful if self.mean_pcr_successful is not None else float("inf")
        return (
            self.success_rate,
            -pcr,
            -self.mean_transition_evaluations,
        )


class QTrainer:
    """Owns online/target R-GNNs and performs relational Q-learning on CUDA."""

    def __init__(
        self,
        config: Config,
        instances: Mapping[int, MazeInstance],
        graph_builder: RelationalGraphBuilder,
        *,
        device: Optional[str | torch.device] = None,
        model: Optional[RelationalQNetwork] = None,
        target_model: Optional[RelationalQNetwork] = None,
    ) -> None:
        self.config = config
        self.instances = dict(instances)
        self.graph_builder = graph_builder
        selected = torch.device(config.device if device is None else device)
        if selected.type != "cuda":
            raise RuntimeError("QTrainer refuses non-CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for R-GNN training. Install a CUDA-enabled PyTorch build."
            )
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device does not exist: {selected}")
        self.device = selected

        self.model = model if model is not None else build_rgnn(config)
        self.model.to(self.device)
        self.target_model = target_model if target_model is not None else build_rgnn(config)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

        readout_ids = {id(parameter) for parameter in self.model.q_readout.parameters()}
        body_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in readout_ids
        ]
        readout_parameters = list(self.model.q_readout.parameters())
        if not body_parameters or not readout_parameters:
            raise RuntimeError("R-GNN body and readout parameter groups must be non-empty")
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": body_parameters,
                    "lr": config.body_learning_rate,
                },
                {
                    "params": readout_parameters,
                    "lr": config.q_readout_learning_rate,
                },
            ],
            weight_decay=0.0,
        )
        self.gradient_updates = 0
        self.target_updates = 0
        self.last_device_report: Optional[Dict[str, str]] = None
        self._verify_cuda_models()

    def _verify_cuda_models(self) -> None:
        for name, model in (("online", self.model), ("target", self.target_model)):
            devices = {parameter.device for parameter in model.parameters()}
            if not devices or any(device.type != "cuda" for device in devices):
                raise RuntimeError(f"{name} R-GNN parameters are not entirely on CUDA")
            if any(device != self.device for device in devices):
                raise RuntimeError(
                    f"{name} R-GNN parameters are on {devices}, expected {self.device}"
                )

    def _instance(self, instance_id: int) -> MazeInstance:
        try:
            return self.instances[int(instance_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown replay instance_id {instance_id}") from exc

    def _build_state_batch(
        self,
        transitions: Sequence[TransitionExperience],
        *,
        next_state: bool,
    ) -> RelationalGraphBatch:
        graphs = []
        for transition in transitions:
            instance = self._instance(transition.instance_id)
            if transition.goal != instance.goal:
                raise ValueError(
                    f"Transition goal {transition.goal} does not match instance {instance.goal}"
                )
            position = transition.next_position if next_state else transition.position
            graphs.append(self.graph_builder.build(instance, position))
        return batch_relational_graphs(graphs, device=self.device)

    def _verify_graph_batch_device(self, batch: RelationalGraphBatch) -> None:
        if batch.device != self.device:
            raise RuntimeError(f"Graph batch is on {batch.device}, expected {self.device}")
        for predicate, tensor in batch.relations.items():
            if tensor.device != self.device:
                raise RuntimeError(f"Relation {predicate} is not on {self.device}")
        if batch.action_object_indices.device != self.device:
            raise RuntimeError("Action object tensor is not on CUDA")

    def compute_targets(
        self,
        transitions: Sequence[TransitionExperience],
    ) -> torch.Tensor:
        """Bellman/search-bound targets using only legal successor actions."""
        if not transitions:
            raise ValueError("Empty transition batch")
        next_batch = self._build_state_batch(transitions, next_state=True)
        self._verify_graph_batch_device(next_batch)
        self.target_model.eval()
        with torch.no_grad():
            next_output = self.target_model(next_batch)
            max_next_q = next_batch.action_max_per_graph(next_output.q_values)

        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        return_bounds = torch.as_tensor(
            [transition.return_bound for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        terminal = torch.as_tensor(
            [transition.terminal for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        dead_end = torch.as_tensor(
            [transition.dead_end for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        active_bootstrap = (~terminal) & (~dead_end)
        if bool((active_bootstrap & (~torch.isfinite(max_next_q))).any()):
            raise RuntimeError(
                "Non-terminal transition has no legal successor action in the target graph"
            )
        safe_max = torch.where(
            torch.isfinite(max_next_q),
            max_next_q,
            torch.zeros_like(max_next_q),
        )
        bellman_hat = rewards + self.config.gamma * safe_max

        if self.config.use_solution_bounds:
            finite_bound = torch.isfinite(return_bounds)
            target = torch.where(
                finite_bound,
                torch.maximum(return_bounds, bellman_hat),
                bellman_hat,
            )
        else:
            target = bellman_hat

        target = torch.where(terminal, rewards, target)
        target = torch.where(
            dead_end,
            torch.full_like(target, float(self.config.dead_end_return)),
            target,
        )
        return target

    def train_step(self, transitions: Sequence[TransitionExperience]) -> UpdateStats:
        if not transitions:
            raise ValueError("Cannot train on an empty batch")
        self.model.train()
        self.target_model.eval()
        self._verify_cuda_models()

        current_batch = self._build_state_batch(transitions, next_state=False)
        self._verify_graph_batch_device(current_batch)
        self.last_device_report = self.device_report(current_batch)
        target = self.compute_targets(transitions)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(current_batch)
        selected_indices = current_batch.select_action_indices(
            [transition.action for transition in transitions]
        )
        if output.q_values.numel() == 0:
            raise RuntimeError("Online R-GNN produced no action Q values")
        chosen_q = output.q_values.gather(0, selected_indices)
        loss = F.mse_loss(chosen_q, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        detached_q = output.q_values.detach()
        q_mean = float(detached_q.mean().item())
        q_min = float(detached_q.min().item())
        q_max = float(detached_q.max().item())

        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm,
            ).item()
        )
        self.optimizer.step()

        self.gradient_updates += 1
        target_updated = False
        if self.gradient_updates % self.config.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            for parameter in self.target_model.parameters():
                parameter.requires_grad_(False)
            self.target_updates += 1
            target_updated = True

        return UpdateStats(
            loss=float(loss.item()),
            q_mean=q_mean,
            q_min=q_min,
            q_max=q_max,
            gradient_norm=gradient_norm,
            target_network_updated=target_updated,
            gradient_updates=self.gradient_updates,
            num_action_objects=int(output.q_values.numel()),
        )

    def device_report(self, batch: RelationalGraphBatch) -> Dict[str, str]:
        return {
            "online_parameters": str(next(self.model.parameters()).device),
            "target_parameters": str(next(self.target_model.parameters()).device),
            "graph_relations": str(batch.relations["CURRENT"].device),
            "q_input_objects": str(batch.action_object_indices.device),
        }


class InstancePoolSampler:
    """Mutually exclusive unsolved/solved/satisficed weighted sampler."""

    def __init__(self, config: Config, instance_ids: Sequence[int], seed: int) -> None:
        if not instance_ids:
            raise ValueError("At least one training instance is required")
        self.config = config
        self.instance_ids = [int(instance_id) for instance_id in instance_ids]
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("Instance IDs must be unique")
        self._pools: Dict[str, Set[int]] = {name: set() for name in INSTANCE_CATEGORIES}
        self._pools[CATEGORY_UNSOLVED] = set(self.instance_ids)
        self._rng = random.Random(int(seed))
        self._weights = {
            CATEGORY_UNSOLVED: float(config.unsolved_weight),
            CATEGORY_SOLVED: float(config.solved_weight),
            CATEGORY_SATISFICED: float(config.satisficed_weight),
        }

    @staticmethod
    def classify(result: SearchResult) -> str:
        if not result.success:
            return CATEGORY_UNSOLVED
        if result.node_expansions == result.plan_length:
            return CATEGORY_SOLVED
        return CATEGORY_SATISFICED

    def update(self, instance_id: int, category: str) -> None:
        instance_id = int(instance_id)
        if category not in INSTANCE_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if not any(instance_id in pool for pool in self._pools.values()):
            raise ValueError(f"Unknown instance id {instance_id}")
        for pool in self._pools.values():
            pool.discard(instance_id)
        self._pools[category].add(instance_id)

    def choose(self) -> int:
        if not self.config.use_instance_sampling:
            return self._rng.choice(self.instance_ids)
        available = [
            (name, self._weights[name], sorted(pool))
            for name, pool in self._pools.items()
            if pool and self._weights[name] > 0
        ]
        if not available:
            raise RuntimeError("No non-empty pool with positive sampling weight")
        total = sum(weight for _, weight, _ in available)
        draw = self._rng.random() * total
        cumulative = 0.0
        selected = available[-1]
        for candidate in available:
            cumulative += candidate[1]
            if draw <= cumulative:
                selected = candidate
                break
        return self._rng.choice(selected[2])

    def counts(self) -> Dict[str, int]:
        return {name: len(pool) for name, pool in self._pools.items()}

    def category_of(self, instance_id: int) -> str:
        for name, pool in self._pools.items():
            if int(instance_id) in pool:
                return name
        raise KeyError(instance_id)


@torch.inference_mode()
def validate_network(
    network: RelationalQNetwork,
    instances: Sequence[MazeInstance],
    config: Config,
    device: str | torch.device,
    graph_builder: Optional[RelationalGraphBuilder] = None,
) -> ValidationMetrics:
    """Evaluate the current R-GNN-guided search with no updates or replay writes."""
    if not instances:
        raise ValueError("Validation requires at least one instance")
    builder = graph_builder if graph_builder is not None else RelationalGraphBuilder(config)
    search = QGuidedWAStar(network, builder, config, device)
    network.eval()
    solved = 0
    pcr_values: List[float] = []
    path_costs: List[float] = []
    unique_expansions = 0
    reexpansions = 0
    transition_evaluations = 0
    for instance in instances:
        result = search.search(instance)
        unique_expansions += result.unique_state_expansions
        reexpansions += result.reexpansions
        transition_evaluations += result.transition_evaluations
        if result.success:
            solved += 1
            if result.path_cost is None:
                raise RuntimeError("Successful search has no path cost")
            path_costs.append(float(result.path_cost))
            if instance.optimal_cost is None or instance.optimal_cost <= 0:
                raise RuntimeError(
                    f"Instance {instance.instance_id} lacks a positive Dijkstra reference cost"
                )
            pcr_values.append(float(result.path_cost) / float(instance.optimal_cost))
    count = len(instances)
    return ValidationMetrics(
        num_instances=count,
        num_solved=solved,
        success_rate=100.0 * solved / count,
        mean_pcr_successful=(sum(pcr_values) / len(pcr_values)) if pcr_values else None,
        mean_path_cost_successful=(
            sum(path_costs) / len(path_costs) if path_costs else None
        ),
        mean_unique_state_expansions=unique_expansions / count,
        mean_reexpansions=reexpansions / count,
        mean_transition_evaluations=transition_evaluations / count,
    )