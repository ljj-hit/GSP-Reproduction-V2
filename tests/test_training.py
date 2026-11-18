"""Q-learning, CUDA, Replay, validation and fixed metric tests for V2."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
import torch

from check_cuda import require_cuda
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
from metrics import summarize_by_structure, summarize_records
from model import build_rgnn
from relational_graph import RelationalGraphBuilder, batch_relational_graphs
from replay_buffer import ReplayBuffer
from search import QGuidedWAStar, SearchResult, TransitionExperience
from trainer import (
    CATEGORY_SATISFICED,
    CATEGORY_SOLVED,
    CATEGORY_UNSOLVED,
    InstancePoolSampler,
    QTrainer,
    validate_network,
)
from utils import set_seed


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="V2 formal training requires an NVIDIA CUDA GPU",
)
DEVICE = "cuda:0"


def make_instance(instance_id: int = 0) -> MazeInstance:
    grid = np.full((16, 16), TERRAIN_1, dtype=np.int8)
    # One narrow next state with a single legal successor.
    grid[7, 8] = WALL
    grid[9, 8] = WALL
    grid[8, 9] = WALL
    start, goal = (7, 7), (15, 15)
    shortest = dijkstra_optimal_path(grid, start, goal)
    counts, ratios = terrain_statistics(grid)
    return MazeInstance(
        terrain_grid=grid,
        start=start,
        goal=goal,
        instance_id=instance_id,
        structure="random_obstacle",
        seed=11,
        map_hash=terrain_map_hash(grid),
        bfs_steps=bfs_shortest_steps(grid, start, goal),
        optimal_cost=shortest.path_cost,
        terrain_counts=counts,
        terrain_ratios=ratios,
    )


def make_transitions(instance: MazeInstance) -> List[TransitionExperience]:
    goal = instance.goal
    return [
        # Normal transition; next state (8,8) has only LEFT legal.
        TransitionExperience(
            instance_id=instance.instance_id,
            position=(8, 7),
            goal=goal,
            action=3,
            next_position=(8, 8),
            reward=-1.0,
            return_bound=-math.inf,
            terminal=False,
            dead_end=False,
        ),
        # Terminal transition.
        TransitionExperience(
            instance_id=instance.instance_id,
            position=(15, 14),
            goal=goal,
            action=3,
            next_position=(15, 15),
            reward=-1.0,
            return_bound=-1.0,
            terminal=True,
            dead_end=False,
        ),
        # Artificial dead-end flag tests the target branch (the environment
        # itself remains connected and does not fabricate dead ends).
        TransitionExperience(
            instance_id=instance.instance_id,
            position=(7, 7),
            goal=goal,
            action=0,
            next_position=(6, 7),
            reward=-1.0,
            return_bound=-2048.0,
            terminal=False,
            dead_end=True,
        ),
        # Finite solution lower bound on a non-terminal non-dead edge.
        TransitionExperience(
            instance_id=instance.instance_id,
            position=(8, 7),
            goal=goal,
            action=1,
            next_position=(9, 7),
            reward=-1.0,
            return_bound=-0.5,
            terminal=False,
            dead_end=False,
        ),
    ]


def build_trainer(config: Config, instance: MazeInstance):
    set_seed(config.training_seed)
    model = build_rgnn(config, device=DEVICE)
    builder = RelationalGraphBuilder(config)
    trainer = QTrainer(
        config,
        {instance.instance_id: instance},
        builder,
        device=DEVICE,
        model=model,
    )
    return trainer, builder, model


def test_cuda_is_really_used_and_cpu_is_rejected() -> None:
    require_cuda(DEVICE, verbose=False)
    config = Config()
    instance = make_instance()
    trainer, builder, model = build_trainer(config, instance)
    assert next(trainer.model.parameters()).device.type == "cuda"
    assert next(trainer.target_model.parameters()).device.type == "cuda"
    assert trainer.optimizer.param_groups[0]["lr"] == config.body_learning_rate
    assert trainer.optimizer.param_groups[1]["lr"] == config.q_readout_learning_rate
    assert all(not parameter.requires_grad for parameter in trainer.target_model.parameters())
    with pytest.raises(RuntimeError):
        QTrainer(config, {instance.instance_id: instance}, builder, device="cpu")


def test_weighted_bellman_targets_terminal_dead_end_and_solution_bounds() -> None:
    config = Config()
    instance = make_instance()
    trainer, builder, _ = build_trainer(config, instance)
    transitions = make_transitions(instance)

    next_graphs = [builder.build(instance, transition.next_position) for transition in transitions]
    from environment import valid_actions_for

    for graph, transition in zip(next_graphs, transitions):
        assert graph.action_ids == tuple(
            valid_actions_for(instance.terrain_grid, transition.next_position)
        )
    next_batch = batch_relational_graphs(next_graphs, device=DEVICE)
    with torch.no_grad():
        next_output = trainer.target_model(next_batch)
        max_next = next_batch.action_max_per_graph(next_output.q_values)
    # The normal next state has exactly one legal action (LEFT); no padding.
    assert next_graphs[0].action_ids == (2,)
    expected = []
    for index, transition in enumerate(transitions):
        if transition.terminal:
            expected.append(transition.reward)
        elif transition.dead_end:
            expected.append(config.dead_end_return)
        else:
            bellman = transition.reward + config.gamma * float(max_next[index].item())
            finite = torch.isfinite(
                torch.tensor(transition.return_bound, dtype=torch.float32)
            ).item()
            expected.append(
                max(transition.return_bound, bellman) if finite else bellman
            )
    actual = trainer.compute_targets(transitions).detach().cpu().numpy()
    assert np.allclose(actual, np.asarray(expected), atol=1e-6)
    assert actual[1] == pytest.approx(transitions[1].reward)  # no bootstrap
    assert actual[2] == pytest.approx(-2048.0)
    assert actual[3] == pytest.approx(max(-0.5, expected[3]))


def test_no_solution_bounds_ablation_uses_bellman_only() -> None:
    config = Config().copy_with(use_solution_bounds=False)
    instance = make_instance()
    trainer, builder, _ = build_trainer(config, instance)
    transitions = make_transitions(instance)
    next_batch = batch_relational_graphs(
        [builder.build(instance, transition.next_position) for transition in transitions],
        device=DEVICE,
    )
    with torch.no_grad():
        max_next = next_batch.action_max_per_graph(
            trainer.target_model(next_batch).q_values
        )
    expected_finite = (
        transitions[3].reward
        + config.gamma * float(max_next[3].item())
    )
    actual = trainer.compute_targets(transitions)
    assert actual[3].item() == pytest.approx(expected_finite, abs=1e-6)
    assert actual[1].item() == pytest.approx(transitions[1].reward)
    assert actual[2].item() == pytest.approx(config.dead_end_return)


def test_online_parameters_update_and_target_syncs_on_schedule() -> None:
    config = Config().copy_with(target_update_interval=2)
    instance = make_instance()
    trainer, _, _ = build_trainer(config, instance)
    transitions = make_transitions(instance)
    online_before = [p.detach().clone() for p in trainer.model.parameters()]
    target_initial = [p.detach().clone() for p in trainer.target_model.parameters()]

    first = trainer.train_step(transitions)
    assert first.gradient_updates == 1
    assert first.target_network_updated is False
    assert math.isfinite(first.loss)
    assert math.isfinite(first.q_mean)
    assert first.q_min <= first.q_max
    assert first.num_action_objects > 0
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(online_before, trainer.model.parameters())
    )
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(target_initial, trainer.target_model.parameters())
    )
    assert all(
        token.startswith("cuda") for token in trainer.last_device_report.values()
    )

    second = trainer.train_step(transitions)
    assert second.target_network_updated is True
    assert trainer.target_updates == 1
    for online, target in zip(
        trainer.model.state_dict().values(),
        trainer.target_model.state_dict().values(),
    ):
        assert torch.equal(online, target)


def test_replay_buffer_compact_fifo_and_sampling() -> None:
    buffer = ReplayBuffer(capacity=3, seed=5)
    instance = make_instance()
    for index, transition in enumerate(make_transitions(instance)):
        transition.return_bound = float(index)
        buffer.add(transition)
    assert len(buffer) == 3
    assert [exp.return_bound for exp in buffer.get_all()] == [1.0, 2.0, 3.0]
    sampled = buffer.sample(2)
    assert len(sampled) == 2
    assert all(isinstance(exp, TransitionExperience) for exp in sampled)
    with pytest.raises(ValueError):
        buffer.sample(4)


def test_instance_pool_sampler_categories_are_mutually_exclusive() -> None:
    config = Config()
    sampler = InstancePoolSampler(config, [0, 1, 2, 3], seed=7)
    success_short = SearchResult(True, actions=[0, 1, 2], node_expansions=3)
    success_long = SearchResult(True, actions=[0, 1, 2], node_expansions=7)
    failure = SearchResult(False, node_expansions=7)
    assert InstancePoolSampler.classify(success_short) == CATEGORY_SOLVED
    assert InstancePoolSampler.classify(success_long) == CATEGORY_SATISFICED
    assert InstancePoolSampler.classify(failure) == CATEGORY_UNSOLVED
    sampler.update(0, CATEGORY_SOLVED)
    sampler.update(1, CATEGORY_SATISFICED)
    assert sampler.counts() == {
        CATEGORY_UNSOLVED: 2,
        CATEGORY_SOLVED: 1,
        CATEGORY_SATISFICED: 1,
    }
    assert sampler.category_of(0) == CATEGORY_SOLVED
    sampler.update(0, CATEGORY_UNSOLVED)
    assert sampler.category_of(0) == CATEGORY_UNSOLVED
    assert all(sampler.choose() in {0, 1, 2, 3} for _ in range(30))


def test_uniform_sampling_ablation_ignores_instance_pools() -> None:
    config = Config().copy_with(use_instance_sampling=False)
    sampler = InstancePoolSampler(config, [10, 11], seed=11)
    sampler.update(10, CATEGORY_SOLVED)
    sampler.update(11, CATEGORY_UNSOLVED)
    draws = {sampler.choose() for _ in range(64)}
    assert draws == {10, 11}


def test_search_replay_train_step_integration() -> None:
    config = Config().copy_with(transition_budget=64, batch_size=8)
    instance = make_instance()
    trainer, builder, model = build_trainer(config, instance)
    search = QGuidedWAStar(model, builder, config, DEVICE)
    result = search.search(instance)
    assert result.experiences
    buffer = ReplayBuffer(128, seed=1)
    buffer.add_many(result.experiences)
    transitions = buffer.sample(min(8, len(buffer)))
    while len(transitions) < 8:
        transitions.append(result.experiences[0])
    stats = trainer.train_step(transitions)
    assert stats.gradient_updates == 1
    assert math.isfinite(stats.loss)


def test_validation_does_not_update_parameters() -> None:
    config = Config().copy_with(transition_budget=16)
    instance = make_instance()
    trainer, builder, model = build_trainer(config, instance)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = validate_network(
        model,
        [instance],
        config,
        DEVICE,
        graph_builder=builder,
    )
    after = [parameter.detach().clone() for parameter in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert metrics.num_instances == 1
    assert 0.0 <= metrics.success_rate <= 100.0
    assert metrics.mean_transition_evaluations <= config.transition_budget
    assert metrics.mean_reexpansions == 0.0


def test_fixed_metrics_success_pcr_and_failure_handling() -> None:
    records = [
        {
            "algorithm": "Trained GSP-RGNN + WA*",
            "structure": "random_obstacle",
            "success": True,
            "path_cost": 15.0,
            "pcr": 1.5,
            "unique_state_expansions": 10,
            "reexpansions": 0,
            "transition_evaluations": 20,
            "search_time_ms": 2.0,
            "gnn_inference_time_ms": 1.0,
            "budget_limited": True,
            "failure_reason": "",
        },
        {
            "algorithm": "Trained GSP-RGNN + WA*",
            "structure": "random_obstacle",
            "success": False,
            "path_cost": None,
            "pcr": None,
            "unique_state_expansions": 30,
            "reexpansions": 0,
            "transition_evaluations": 50,
            "search_time_ms": 3.0,
            "gnn_inference_time_ms": 1.5,
            "budget_limited": True,
            "failure_reason": "transition_budget",
        },
    ]
    summary = summarize_records(records)["Trained GSP-RGNN + WA*"]
    assert summary["success_rate_percent"] == pytest.approx(50.0)
    assert summary["num_solved"] == 1
    assert summary["num_failed"] == 1
    assert summary["mean_pcr_successful"] == pytest.approx(1.5)
    assert summary["mean_path_cost_successful"] == pytest.approx(15.0)
    assert summary["mean_unique_state_expansions"] == pytest.approx(20.0)
    assert summary["mean_transition_evaluations"] == pytest.approx(35.0)
    assert summary["mean_search_time_ms"] == pytest.approx(2.5)


def test_metrics_grouping_by_five_structures() -> None:
    structures = [
        "random_obstacle",
        "rooms_corridors",
        "u_detour",
        "branching_maze",
        "loop_maze",
    ]
    records = []
    for index, structure in enumerate(structures):
        records.append(
            {
                "algorithm": "WA* + Manhattan",
                "structure": structure,
                "success": index < 4,
                "path_cost": 10.0 if index < 4 else None,
                "pcr": 1.0 + index * 0.1 if index < 4 else None,
                "unique_state_expansions": 5 + index,
                "reexpansions": 0,
                "transition_evaluations": 10 + index,
                "search_time_ms": 1.0,
                "gnn_inference_time_ms": 0.0,
                "budget_limited": True,
                "failure_reason": "" if index < 4 else "transition_budget",
            }
        )
    grouped = summarize_by_structure(records)
    assert set(grouped) == set(structures)
    for index, structure in enumerate(structures):
        entry = grouped[structure]["WA* + Manhattan"]
        assert entry["num_instances"] == 1
        assert entry["success_rate_percent"] == (100.0 if index < 4 else 0.0)