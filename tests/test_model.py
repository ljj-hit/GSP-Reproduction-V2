"""R-GNN architecture, SmoothMax, role messages, pooling and permutation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from environment import (
    MazeInstance,
    TERRAIN_1,
    WALL,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    terrain_map_hash,
    terrain_statistics,
)
from model import RelationalQNetwork, build_rgnn, smoothmax_aggregate
from relational_graph import (
    PREDICATE_ARITIES,
    PREDICATES,
    RelationalGraph,
    RelationalGraphBuilder,
    batch_relational_graphs,
    remap_graph_objects,
)
from utils import load_checkpoint, save_checkpoint, set_seed


def make_instance() -> MazeInstance:
    grid = np.full((16, 16), TERRAIN_1, dtype=np.int8)
    grid[2, 3] = WALL
    grid[7, 3] = WALL
    start, goal = (7, 7), (15, 15)
    shortest = dijkstra_optimal_path(grid, start, goal)
    counts, ratios = terrain_statistics(grid)
    return MazeInstance(
        terrain_grid=grid,
        start=start,
        goal=goal,
        instance_id=0,
        structure="test",
        seed=7,
        map_hash=terrain_map_hash(grid),
        bfs_steps=bfs_shortest_steps(grid, start, goal),
        optimal_cost=shortest.path_cost,
        terrain_counts=counts,
        terrain_ratios=ratios,
    )


def empty_fact(predicate: str) -> np.ndarray:
    return np.empty((0, PREDICATE_ARITIES[predicate]), dtype=np.int64)


def synthetic_batch(num_objects: int, predicate: str, fact) -> object:
    relations = {name: empty_fact(name) for name in PREDICATES}
    relations[predicate] = np.asarray([fact], dtype=np.int64)
    graph = RelationalGraph(
        num_objects=num_objects,
        relations=relations,
        cell_object_ids=tuple(range(num_objects)),
        action_object_ids=tuple(),
        action_ids=tuple(),
        current_cell_id=0,
        goal_cell_id=0,
        num_cells=num_objects,
    )
    return batch_relational_graphs([graph], device="cpu")


def test_architecture_dimensions_and_independent_predicate_combiners() -> None:
    config = Config()
    model = build_rgnn(config)
    assert isinstance(model, RelationalQNetwork)
    assert model.embedding_dim == 32
    assert model.gnn_layers == 4
    assert model.hidden_dim == 64
    assert len(model.predicate_combiners) == len(PREDICATES) == 14
    parameter_ids = [
        id(parameter)
        for predicate in PREDICATES
        for parameter in model.predicate_combiners[predicate].parameters()
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert not any(isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)) for module in model.modules())
    first = model.predicate_combiners["UP"].mlp
    second = model.predicate_combiners["DOWN"].mlp
    assert next(first.parameters()).data_ptr() != next(second.parameters()).data_ptr()
    assert model.q_readout[-1].out_features == 1


def test_smoothmax_values_permutation_and_stability() -> None:
    messages = torch.tensor(
        [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    receivers = torch.tensor([0, 0, 0], dtype=torch.long)
    result = smoothmax_aggregate(messages, receivers, 2, 1.0)
    expected = torch.softmax(messages, dim=0)
    values = (expected * messages).sum(dim=0)
    assert torch.allclose(result[0], values)
    assert torch.equal(result[1], torch.zeros(2, dtype=torch.float64))

    permutation = torch.tensor([2, 0, 1])
    permuted = smoothmax_aggregate(messages[permutation], receivers, 2, 1.0)
    assert torch.allclose(result, permuted)
    empty = smoothmax_aggregate(
        torch.empty((0, 2), dtype=torch.float64),
        torch.empty((0,), dtype=torch.long),
        2,
        1.0,
    )
    assert torch.equal(empty, torch.zeros(2, 2, dtype=torch.float64))

    stable_messages = torch.tensor([[1000.0], [-1000.0], [999.0]], dtype=torch.float32)
    stable = smoothmax_aggregate(
        stable_messages,
        torch.zeros(3, dtype=torch.long),
        1,
        50.0,
    )
    assert torch.isfinite(stable).all()
    assert 999.0 <= stable.item() <= 1000.0


def _role_message_test(predicate: str, num_objects: int, fact) -> None:
    config = Config()
    model = build_rgnn(config)
    batch = synthetic_batch(num_objects, predicate, fact)
    rng = torch.Generator().manual_seed(11)
    x = torch.randn((num_objects, config.embedding_dim), generator=rng)
    aggregated = model._messages(batch, x)
    arguments = torch.as_tensor(fact, dtype=torch.long)
    flat = x[arguments].reshape(1, -1)
    combined = model.predicate_combiners[predicate](flat).reshape(-1, config.embedding_dim)
    for role, object_id in enumerate(fact):
        assert torch.allclose(
            aggregated[int(object_id)], combined[role], atol=1e-6, rtol=1e-6
        ), (predicate, role, object_id)
    untouched = [index for index in range(num_objects) if index not in fact]
    for object_id in untouched:
        assert torch.equal(aggregated[object_id], torch.zeros(config.embedding_dim))


def test_binary_predicate_roles_receive_distinct_messages() -> None:
    _role_message_test("UP", 2, (0, 1))
    _role_message_test("MOVE_UP", 3, (2, 0, 1))


def test_residual_update_and_shared_updater() -> None:
    config = Config().copy_with(gnn_layers=1)
    model = build_rgnn(config)
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    batch = batch_relational_graphs([builder.build(instance, (7, 7))], device="cpu")
    x0 = torch.zeros((batch.num_objects, config.embedding_dim))
    groups = model._prepare_message_groups(batch)
    aggregated = model._messages(batch, x0, groups)
    expected = x0 + model.Comb_U(torch.cat((x0, aggregated), dim=-1))
    output = model(batch)
    assert torch.allclose(output.object_embeddings, expected, atol=1e-6, rtol=1e-6)


def test_global_pooling_and_shared_q_readout() -> None:
    config = Config()
    model = build_rgnn(config)
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    graph_a = builder.build(instance, (7, 7))
    graph_b = builder.build(instance, (0, 0))
    batch = batch_relational_graphs([graph_a, graph_b], device="cpu")
    output = model(batch)
    expected_global = smoothmax_aggregate(
        output.object_embeddings,
        batch.object_graph_indices,
        batch.num_graphs,
        config.smoothmax_beta,
    )
    assert torch.allclose(output.global_embeddings, expected_global, atol=1e-6, rtol=1e-6)
    action_embeddings = output.object_embeddings[batch.action_object_indices]
    action_globals = output.global_embeddings[batch.action_graph_indices]
    expected_q = model.q_readout(
        torch.cat((action_embeddings, action_globals), dim=-1)
    ).squeeze(-1)
    assert torch.allclose(output.q_values, expected_q)
    assert output.q_values.shape[0] == len(batch.action_ids)


def test_variable_action_counts_forward_and_mapping() -> None:
    config = Config()
    model = build_rgnn(config)
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    graphs = [
        builder.build(instance, (0, 0)),   # 2 actions
        builder.build(instance, (2, 2)),   # 3 actions
        builder.build(instance, (7, 7)),   # 4 actions
    ]
    batch = batch_relational_graphs(graphs, device="cpu")
    output = model(batch)
    assert batch.graph_action_counts == (2, 3, 4)
    assert output.q_values.shape == (9,)
    assert output.action_ids.tolist() == [
        action for graph in graphs for action in graph.action_ids
    ]


def test_permutation_invariance_of_object_ids() -> None:
    config = Config()
    set_seed(123)
    model = build_rgnn(config)
    model.eval()
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    graph = builder.build(instance, (7, 7))
    rng = np.random.default_rng(99)
    permutation = rng.permutation(graph.num_objects)
    remapped = remap_graph_objects(graph, permutation)
    batch_a = batch_relational_graphs([graph], device="cpu")
    batch_b = batch_relational_graphs([remapped], device="cpu")
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert out_a.action_ids.tolist() == out_b.action_ids.tolist()
    assert torch.allclose(out_a.q_values, out_b.q_values, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        out_a.global_embeddings, out_b.global_embeddings, atol=1e-5, rtol=1e-5
    )


def test_disjoint_batch_does_not_contaminate_each_graph() -> None:
    config = Config()
    model = build_rgnn(config)
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    graph_a = builder.build(instance, (7, 7))
    graph_b = builder.build(instance, (0, 0))
    with torch.no_grad():
        solo_a = model(batch_relational_graphs([graph_a], device="cpu"))
        combined = model(batch_relational_graphs([graph_a, graph_b], device="cpu"))
    count_a = len(graph_a.action_ids)
    assert torch.allclose(solo_a.q_values, combined.q_values[:count_a], atol=1e-6)
    assert torch.allclose(solo_a.global_embeddings[0], combined.global_embeddings[0], atol=1e-6)


def test_checkpoint_round_trip_outputs_identical(tmp_path: Path) -> None:
    config = Config()
    set_seed(321)
    model = build_rgnn(config)
    instance = make_instance()
    builder = RelationalGraphBuilder(config)
    batch = batch_relational_graphs([builder.build(instance, (7, 7))], device="cpu")
    path = tmp_path / "initial_model.pt"
    save_checkpoint(path, model, episode=0, config=config, extra={"kind": "initial"})
    payload = load_checkpoint(path)
    reloaded = build_rgnn(config)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        expected = model(batch).q_values
        actual = reloaded(batch).q_values
    assert torch.equal(expected, actual)
    assert payload["config"]["embedding_dim"] == 32
    assert payload["kind"] == "initial"