"""Predicate-specific relational GNN and shared action Q readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import Config
from relational_graph import (
    PREDICATE_ARITIES,
    PREDICATES,
    RelationalGraphBatch,
)


@dataclass
class RGraphOutput:
    """Flat per-action-object output and graph/action mappings."""

    q_values: torch.Tensor
    action_graph_indices: torch.Tensor
    action_ids: torch.Tensor
    global_embeddings: torch.Tensor
    object_embeddings: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PredicateCombiner(nn.Module):
    """Comb_p: R^{m*d} -> R^{m*d}, split into role-specific messages later."""

    def __init__(self, arity: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.arity = int(arity)
        self.embedding_dim = int(embedding_dim)
        self.mlp = _mlp(arity * embedding_dim, hidden_dim, arity * embedding_dim)

    def forward(self, flat_arguments: torch.Tensor) -> torch.Tensor:
        expected = self.arity * self.embedding_dim
        if flat_arguments.shape[-1] != expected:
            raise ValueError(
                f"Expected flat predicate arguments of size {expected}, "
                f"got {tuple(flat_arguments.shape)}"
            )
        return self.mlp(flat_arguments)


def smoothmax_aggregate(
    messages: torch.Tensor,
    receivers: torch.Tensor,
    num_receivers: int,
    beta: float,
) -> torch.Tensor:
    """Dimension-wise numerically stable SmoothMax per receiver.

    ``SmoothMax_beta(x_1..x_n) = sum(x_i exp(beta x_i)) / sum(exp(beta x_i))``.
    Receivers with no messages receive a zero vector.
    """
    if messages.ndim != 2:
        raise ValueError("Messages must have shape (E, d)")
    if receivers.shape != (messages.shape[0],):
        raise ValueError("Receivers must have shape (E,)")
    if beta <= 0:
        raise ValueError("beta must be positive")
    embedding_dim = messages.shape[1]
    output = torch.zeros(
        (num_receivers, embedding_dim),
        dtype=messages.dtype,
        device=messages.device,
    )
    if messages.numel() == 0:
        return output

    expanded_receivers = receivers.unsqueeze(1).expand(-1, embedding_dim)
    max_per_receiver = torch.full(
        (num_receivers, embedding_dim),
        -float("inf"),
        dtype=messages.dtype,
        device=messages.device,
    )
    max_per_receiver.scatter_reduce_(
        0,
        expanded_receivers,
        messages,
        reduce="amax",
        include_self=True,
    )
    # The shift is a numerical stabiliser only; stop gradients through it so
    # the derivative remains exactly the SmoothMax derivative.
    shift = max_per_receiver.detach()
    shifted = messages - shift[receivers]
    weights = torch.exp(beta * shifted)
    numerator = torch.zeros_like(output)
    denominator = torch.zeros_like(output)
    numerator.index_add_(0, receivers, weights * messages)
    denominator.index_add_(0, receivers, weights)
    has_messages = denominator > 0
    safe_denominator = torch.where(has_messages, denominator, torch.ones_like(denominator))
    output = torch.where(has_messages, numerator / safe_denominator, output)
    return output


class RelationalQNetwork(nn.Module):
    """L-layer predicate-specific R-GNN with a shared scalar Q readout."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = int(config.embedding_dim)
        self.hidden_dim = int(config.message_hidden_dim)
        self.gnn_layers = int(config.gnn_layers)
        self.beta = float(config.smoothmax_beta)

        # Every predicate has its own Comb_p.  They are intentionally not
        # shared, including distinct direction and MOVE_* predicates.
        self.predicate_combiners = nn.ModuleDict(
            {
                predicate: PredicateCombiner(
                    PREDICATE_ARITIES[predicate],
                    self.embedding_dim,
                    self.hidden_dim,
                )
                for predicate in PREDICATES
            }
        )
        # Comb_U is shared by every object.
        self.update_network = _mlp(
            2 * self.embedding_dim,
            self.hidden_dim,
            self.embedding_dim,
        )
        # One scalar readout shared by every action object.
        self.q_readout = _mlp(
            2 * self.embedding_dim,
            int(config.q_readout_hidden_dim),
            1,
        )

    # Backwards-friendly aliases used in the report/documentation.
    @property
    def Comb_p(self) -> nn.ModuleDict:
        return self.predicate_combiners

    @property
    def Comb_U(self) -> nn.Sequential:
        return self.update_network

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_batch_device(self, batch: RelationalGraphBatch) -> None:
        expected = self._parameter_device()
        for predicate, facts in batch.relations.items():
            if facts.device != expected:
                raise RuntimeError(
                    f"Relation tensor {predicate} is on {facts.device}, "
                    f"but the network is on {expected}"
                )
        if batch.action_object_indices.device != expected:
            raise RuntimeError("Action object indices and network device differ")

    def _predicates_by_arity(self) -> Dict[int, Tuple[str, ...]]:
        groups: Dict[int, List[str]] = {}
        for predicate in PREDICATES:
            groups.setdefault(PREDICATE_ARITIES[predicate], []).append(predicate)
        return {arity: tuple(names) for arity, names in groups.items()}

    def _prepare_message_groups(
        self,
        batch: RelationalGraphBatch,
    ) -> List[Dict[str, object]]:
        """Prepack predicate facts and stack independent Comb_p weights.

        Predicate parameters remain separate.  The packing allows one batched
        linear operation per MLP layer for all predicates of equal arity,
        avoiding hundreds of tiny GPU launches per state.
        """
        groups: List[Dict[str, object]] = []
        device = self._parameter_device()
        for arity, predicates in self._predicates_by_arity().items():
            fact_rows = []
            max_facts = 0
            for predicate in predicates:
                facts = batch.relations[predicate]
                if facts.shape[1] != arity:
                    raise RuntimeError(f"Bad arity tensor for {predicate}")
                fact_rows.append((predicate, facts))
                max_facts = max(max_facts, facts.shape[0])
            if max_facts == 0:
                continue
            padded_object_ids = torch.zeros(
                (len(predicates), max_facts, arity),
                dtype=torch.long,
                device=device,
            )
            active_mask = torch.zeros(
                (len(predicates), max_facts, arity),
                dtype=torch.bool,
                device=device,
            )
            for index, (_, facts) in enumerate(fact_rows):
                count = facts.shape[0]
                if count:
                    padded_object_ids[index, :count] = facts
                    active_mask[index, :count, :] = True
            flat_mask = active_mask.reshape(-1)
            receivers = padded_object_ids.reshape(-1)[flat_mask]
            first_layers = [self.predicate_combiners[p].mlp[0] for p in predicates]
            second_layers = [self.predicate_combiners[p].mlp[2] for p in predicates]
            groups.append(
                {
                    "arity": arity,
                    "padded_object_ids": padded_object_ids,
                    "active_mask": flat_mask,
                    "receivers": receivers,
                    "weight1": torch.stack([layer.weight for layer in first_layers], dim=0),
                    "bias1": torch.stack([layer.bias for layer in first_layers], dim=0),
                    "weight2": torch.stack([layer.weight for layer in second_layers], dim=0),
                    "bias2": torch.stack([layer.bias for layer in second_layers], dim=0),
                }
            )
        return groups

    def _messages(
        self,
        batch: RelationalGraphBatch,
        object_embeddings: torch.Tensor,
        groups: Optional[List[Dict[str, object]]] = None,
    ) -> torch.Tensor:
        num_objects = batch.num_objects
        groups = groups if groups is not None else self._prepare_message_groups(batch)
        receiver_chunks = []
        message_chunks = []
        for group in groups:
            arity = int(group["arity"])
            padded_object_ids = group["padded_object_ids"]
            active_mask = group["active_mask"]
            weight1 = group["weight1"]
            bias1 = group["bias1"]
            weight2 = group["weight2"]
            bias2 = group["bias2"]
            embedded = object_embeddings[padded_object_ids]
            padded = embedded.reshape(
                padded_object_ids.shape[0],
                padded_object_ids.shape[1],
                arity * self.embedding_dim,
            )
            hidden = torch.relu(
                torch.baddbmm(bias1.unsqueeze(1), padded, weight1.transpose(1, 2))
            )
            combined = torch.baddbmm(
                bias2.unsqueeze(1), hidden, weight2.transpose(1, 2)
            )
            messages = combined.reshape(-1, self.embedding_dim)[active_mask]
            message_chunks.append(messages)
            receiver_chunks.append(group["receivers"])
        if not message_chunks:
            return torch.zeros(
                (num_objects, self.embedding_dim),
                dtype=object_embeddings.dtype,
                device=object_embeddings.device,
            )
        receivers = torch.cat(receiver_chunks, dim=0)
        messages = torch.cat(message_chunks, dim=0)
        return smoothmax_aggregate(
            messages,
            receivers,
            num_objects,
            self.beta,
        )

    def forward(self, batch: RelationalGraphBatch) -> RGraphOutput:
        self._validate_batch_device(batch)
        device = self._parameter_device()
        # X_0(o) = 0 for every Cell and Action object.
        object_embeddings = torch.zeros(
            (batch.num_objects, self.embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        message_groups = self._prepare_message_groups(batch)
        for _ in range(self.gnn_layers):
            aggregated = self._messages(batch, object_embeddings, message_groups)
            update = self.update_network(
                torch.cat((object_embeddings, aggregated), dim=-1)
            )
            object_embeddings = object_embeddings + update

        global_embeddings = smoothmax_aggregate(
            object_embeddings,
            batch.object_graph_indices,
            batch.num_graphs,
            self.beta,
        )
        action_embeddings = object_embeddings[batch.action_object_indices]
        action_globals = global_embeddings[batch.action_graph_indices]
        q_values = self.q_readout(
            torch.cat((action_embeddings, action_globals), dim=-1)
        ).squeeze(-1)
        return RGraphOutput(
            q_values=q_values,
            action_graph_indices=batch.action_graph_indices,
            action_ids=batch.action_ids,
            global_embeddings=global_embeddings,
            object_embeddings=object_embeddings,
        )

    def get_q_values(self, batch: RelationalGraphBatch) -> torch.Tensor:
        return self.forward(batch).q_values

    @property
    def readout_parameters(self) -> Sequence[nn.Parameter]:
        return list(self.q_readout.parameters())

    @property
    def body_parameters(self) -> Sequence[nn.Parameter]:
        readout_ids = {id(parameter) for parameter in self.q_readout.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in readout_ids
        ]


def build_rgnn(config: Config, *, device: Optional[str | torch.device] = None) -> RelationalQNetwork:
    model = RelationalQNetwork(config)
    if device is not None:
        model = model.to(torch.device(device))
    return model