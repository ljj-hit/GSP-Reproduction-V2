# GSP-Reproduction-V2

一个使用 **Python + PyTorch** 完成的 GSP（Generalized Search for Planning）机制级简化复现项目（第二版）。

V2 将环境从 8×8 二值迷宫升级为 **16×16 加权地形迷宫**（墙壁 `0`，地形代价 `1/3/5`），并把 Q 函数从 MLP 替换为 **谓词特定的 Relational Graph Neural Network（R-GNN）**。搜索仍采用 GSP 的 **状态—动作 Weighted A\***；训练仍采用 **Q-learning + Replay Buffer + Target Network + 搜索回报下界**。最终在从未参与训练和模型选择的 300 张测试迷宫上比较 5 种固定算法。

> 这是一个**机制级简化复现**，不是论文原始代码的等价实现。R-GNN 按本任务书描述的谓词消息传递机制独立实现，训练配置包含工程选择。主要差异见[第 2 节](#2-与原论文的区别)。

---

## 1. 项目目标

项目验证的核心问题不是“如何用 Dijkstra 解迷宫”，而是：

1. GSP-WA* 搜索产生的状态—动作经验，能否训练出一个跨迷宫布局泛化的 Q 函数；
2. 该 Q 函数能否作为搜索优先级的组成部分，在**相同状态转移评估预算**下改变并改善后续 WA* 的搜索行为；
3. 关系图网络能否在不接收固定格子 ID、行号、列号或结构标签的情况下，仅凭关系事实理解 16×16 空间布局；
4. 整套 `搜索 → 经验 → Q-learning → 更好的搜索` 闭环能否在完全独立的测试集上被定量评估。

训练、验证、测试迷宫全部程序生成，并通过地形布局哈希保证三个集合互不重复。测试集只用于最终评估，不参与梯度更新、模型选择或超参数调整。

---

## 2. 与原论文的区别

| 方面 | 原论文 | 本项目 |
|---|---|---|
| 状态表示/网络 | Relational Graph Neural Network | 按任务书机制实现的 R-GNN（谓词特定消息传递 + SmoothMax + 残差 + 共享 Q Readout） |
| 规划领域 | PDDL 规划领域 | 固定 16×16 四方向加权网格 |
| 泛化范围 | 关系结构、规模与实例泛化 | 仅验证同一 16×16 尺寸内不同地图布局的跨实例泛化 |
| 数据 | PDDL 领域数据/生成任务 | 完全程序生成的迷宫，无外部数据集 |
| 训练稳定化 | 论文原始训练配置 | 额外使用 Target Network；这是工程实现选择 |
| 死路回报下界 | 论文搜索定义与参数 | 固定 `-2048`，低于 16×16 任意简单路径的回报下界（最坏 `-1275`） |
| 实例采样 | 论文的实例选择机制 | Unsolved/Solved/Satisficed 三池 `1:2:4` 加权采样（本项目工程权重，非论文原值） |
| R-GNN 层数 | 论文原始配置 | 默认 `L=4` 的工程初始值，不代表论文规定 |
| 传统算法预算 | 论文实验配置 | A*/WA*/R-GNN 统一 512 次状态转移评估；Dijkstra 不受预算限制 |

此外：

- 本项目**不声称实现跨地图尺寸泛化**。修改 `height/width` 会改变图的规模与数据分布，必须重新生成数据和重训。
- 搜索回报下界来自**搜索实际找到的成功路径**，不是 BFS 最短路径，也不是预先计算的最优 Q 值。
- Dijkstra 最优成本只用于评估参考，**从不作为 R-GNN 输入或 Q-learning 标签**。
- GSP-WA* 按**状态—动作对**扩展，传统 A*/WA* 按**状态**扩展。所有算法使用相同的 512 次转移评估预算，但节点语义不同，解释时必须区分。
- 逐谓词 Comb_p 的等元数批处理执行只是为了减少 GPU 小算子开销；每个谓词的参数仍然独立，算法定义未改变。

---

## 3. 加权迷宫环境

### 3.1 地形与奖励

地形矩阵为 `int8`，取值固定为：

```text
0 = 墙壁，不可进入
1 = 代价 1 地形
3 = 代价 3 地形
5 = 代价 5 地形
```

进入目标格的成本等于该格地形值：

\[
c(s,a,s') = \text{terrain}[s']
\]

奖励：

\[
r(s,a,s') = -c(s,a,s')
\]

| 进入地形 | 奖励 |
|---|---:|
| 1 | -1 |
| 3 | -3 |
| 5 | -5 |
| 墙壁 | 不允许进入 |

起点地形成本不计入路径成本，终点地形成本计入。折扣因子 \(\gamma=1\)。优化目标是累计地形代价最低：

\[
\min_\pi \sum_{t=0}^{T-1} c(s_t,a_t,s_{t+1}).
\]

### 3.2 动作与合法性

动作编号全局一致：

```text
UP = 0    (-1, 0)
DOWN = 1  ( 1, 0)
LEFT = 2  ( 0,-1)
RIGHT = 3 ( 0, 1)
```

只有同时满足“目标格在边界内”和“目标格不是墙壁”的动作才合法。非法动作不会被执行、不会产生经验、不会原地停留。`MazeEnvironment.step()` 对非法动作抛出 `InvalidActionError`。

### 3.3 实例字段

每张地图保存完整地形矩阵、`start=(row,col)`、`goal=(row,col)`、随机种子、结构类型、BFS 最短步数、Dijkstra 最优累计成本、三档地形的数量与比例、地形布局哈希和结构生成元数据。

BFS 与 Dijkstra 只在数据生成和评估阶段使用，不属于网络输入。

---

## 4. 数据集生成

### 4.1 五种结构

[`generator.py`](generator.py) 生成五种结构，结构标签只用于评估分组，**从不作为 R-GNN 输入**：

| 结构 | 生成规则 |
|---|---|
| `random_obstacle` | 随机设置 20%–30% 墙壁，BFS 检查连通性 |
| `rooms_corridors` | 用十字墙线划分四个房间，随机开门洞并加入狭窄内墙 |
| `u_detour` | 生成 U 型屏障，要求起点到终点的 BFS 路径比 Manhattan 距离多至少 4 步，再做旋转/翻转 |
| `branching_maze` | 在奇数坐标格上用随机 DFS（Recursive Backtracking）生成完美迷宫，含多个分支与死胡同 |
| `loop_maze` | 在分支迷宫基础上随机打通 15%–30% 的候选墙，形成环路与替代路线 |

五种结构在每个 split 中数量精确均衡，三个 split 使用不同的 master seed。

### 4.2 正式数据规模

| 数据集 | 总数 | 每种结构 |
|---|---:|---:|
| Train | 1000 | 200 |
| Validation | 200 | 40 |
| Test | 300 | 60 |

`config.py` 中的 `num_train_instances`、`num_validation_instances`、`num_test_instances` 必须能被 5 整除，`Config.validate()` 会检查。

### 4.3 地形分配

墙壁与结构生成完成后，对所有可通行格独立均匀分配地形：

\[
P(c=1)=P(c=3)=P(c=5)=\frac13.
\]

地形分配与结构类型完全独立，不允许某类结构固定使用某种地形。单张地图中某一地形偶然缺失是允许的。

### 4.4 数据有效性

每张地图必须满足：

1. 起点、终点均可通行且不同；
2. 起点到终点 BFS 连通；
3. 符合对应结构的生成规则；
4. 保存 BFS 最短步数与 Dijkstra 最优累计成本；
5. 记录三档地形的数量、比例、随机种子、结构元数据和 SHA-256 地形哈希；
6. Train/Validation/Test 之间无重复地形布局；
7. 生成过程有最大尝试次数，失败时抛出明确异常，不会静默少生成。

### 4.5 保存格式

正式数据目录：

```text
data/16x16_rgnn/
├── train.npz
├── validation.npz
├── test.npz
└── dataset_manifest.json
```

NPZ 数组：

```text
terrains        int8   [N, 16, 16]
starts          int16  [N, 2]
goals           int16  [N, 2]
seeds           int64  [N]
structures      <U24   [N]
bfs_steps       int16  [N]
optimal_costs   float32[N]
map_hashes      <U64   [N]
terrain_counts  int32  [N, 3]
terrain_ratios  float32[N, 3]
```

Manifest 记录实际数量、五结构计数、生成尝试/拒绝统计、地形比例、BFS 步数、最优成本统计和 split 独立性审计。哈希仅覆盖地形布局，不包含起终点，因而比只比较起终点更严格。

> 正式数据集 `data/16x16_rgnn/`（1000/200/300）已生成；另保留用于冒烟验证的小数据集 `.tmp_smoke/data`（100/20/20）。

---

## 5. 关系图结构与谓词

### 5.1 对象集合

对 16×16 地图：

- **Cell Objects**：全部 256 个格子都是对象，墙壁格子保留并用 `WALL` 谓词标记；
- **Action Objects**：当前状态下每个合法动作创建一个对象，数量为 1–4，不固定填充到 4 个；
- 当前格与目标格不是新对象，而是通过 `CURRENT(cell)`、`GOAL(cell)` 关系标记；
- 完整对象集合 \(\widetilde O = O_C \cup O_A\)。

所有对象初始嵌入为零向量：

\[
X_0(o)=0,\quad o\in\widetilde O,\quad d=32.
\]

网络不接收格子 ID、行号、列号、One-hot 位置或结构标签。对象编号只是批处理索引，结构信息全部来自关系事实。

### 5.2 关系谓词（共 14 种）

| 类型 | 谓词 | 元数 | 含义 |
|---|---|---:|---|
| 地形属性 | `WALL(cell)` | 1 | 墙壁 |
| 地形属性 | `TERRAIN_1(cell)` | 1 | 代价 1 |
| 地形属性 | `TERRAIN_3(cell)` | 1 | 代价 3 |
| 地形属性 | `TERRAIN_5(cell)` | 1 | 代价 5 |
| 状态标记 | `CURRENT(cell)` | 1 | 当前格（恰好 1 条） |
| 状态标记 | `GOAL(cell)` | 1 | 目标格（恰好 1 条） |
| 空间关系 | `UP(cell_i, cell_j)` | 2 | 上邻接（有向） |
| 空间关系 | `DOWN(cell_i, cell_j)` | 2 | 下邻接（有向） |
| 空间关系 | `LEFT(cell_i, cell_j)` | 2 | 左邻接（有向） |
| 空间关系 | `RIGHT(cell_i, cell_j)` | 2 | 右邻接（有向） |
| 动作关系 | `MOVE_UP(action, source, target)` | 3 | 合法上移动作 |
| 动作关系 | `MOVE_DOWN(action, source, target)` | 3 | 合法下移动作 |
| 动作关系 | `MOVE_LEFT(action, source, target)` | 3 | 合法左移动作 |
| 动作关系 | `MOVE_RIGHT(action, source, target)` | 3 | 合法右移动作 |

约束：

- 每个格子恰好属于 `WALL / TERRAIN_1 / TERRAIN_3 / TERRAIN_5` 之一；
- 每张图恰好一条 `CURRENT`、一条 `GOAL`；
- 空间关系保留方向，不合并为无向邻接；墙壁格子之间也保留几何邻接；
- 只为合法动作创建 `MOVE_*` 事实，非法动作不进入图。

### 5.3 Action Objects

对当前状态 \(s\) 的合法动作集合 \(A(s)=\{a_1,\ldots,a_k\}\)：

```text
action_object_id = 256 + local_index
MOVE_<NAME>(action_object, current_cell, target_cell)
```

动作对象与动作编号的映射保存在 `action_object_ids` 和 `action_ids` 中。Q 网络对每个动作对象输出一个标量，因此动作数量可变时仍能统一前向；不同动作共用同一个 Q Readout，不建立四套动作网络。

### 5.4 静态缓存与批处理

- **静态图** `StaticMapGraph`：地图谓词、地形谓词、方向邻接、`GOAL`，按 `(map_hash, shape, goal)` 缓存，每张地图只构建一次；
- **动态图** `RelationalGraph`：复制静态关系，加入当前 `CURRENT` 和本状态动作事实；通过有界 LRU 缓存按 `(地图, 位置)` 复用；
- **disjoint-union batching**：多图批处理时，第 \(i\) 张图的对象 ID 加上该图的全局偏移，关系事实不会跨图连接；
- `RelationalGraphBatch` 保存每个动作对象的全局索引、所属图索引、动作编号和图级动作偏移，支持按图取动作最大值；
- `remap_graph_objects` 支持对象重编号并同步重映射全部关系，用于置换不变性测试。

---

## 6. R-GNN 网络

### 6.1 结构参数

| 参数 | 默认值 |
|---|---:|
| 对象嵌入维度 \(d\) | 32 |
| 消息传递层数 \(L\) | 4 |
| Comb_p / Comb_U 隐藏维度 | 64 |
| SmoothMax \(\beta\) | 1.0 |
| Q Readout 隐藏维度 | 64 |
| Q 输出 | 每个动作对象 1 个标量 |

### 6.2 逐谓词消息网络 Comb_p

每个谓词 \(p\) 有**独立的** `PredicateCombiner`，参数不共享。对一条 \(m\) 元事实：

\[
p(o_1,\ldots,o_m),
\]

输入为所有参数对象当前嵌入的拼接：

\[
X_i(o_1)\Vert\cdots\Vert X_i(o_m)\in\mathbb R^{md}.
\]

Comb_p 是一个两层 MLP：

\[
\mathbb R^{md}\rightarrow\mathbb R^{64}\rightarrow\mathbb R^{md}.
\]

输出按参数位置拆分成 \(m\) 个 \(d\) 维消息，每个参数对象收到**与自己角色对应**的消息。例如：

- `UP(c_i,c_j)`：\(c_i\) 与 \(c_j\) 收到不同消息；
- `MOVE_UP(o_a,c_i,c_j)`：动作对象、源格、目标格分别收到不同的 32 维消息。

实现上，元数相同的谓词（一元 6 个、二元 4 个、三元 4 个）在每次前向时把各自独立的权重堆叠成 batch，用 `baddbmm` 一次执行多层线性运算。这只是算子批处理，`Comb_p` 的参数仍逐谓词独立。

### 6.3 SmoothMax 聚合

对每个对象 \(o\)，收集所有包含它的关系事实发给它的消息：

\[
M(o)=\{m_o^q:q\in R,\ o\in q\}.
\]

对每个特征维度分别执行：

\[
\operatorname{SmoothMax}_\beta(x_1,\ldots,x_n)
=
\frac{\sum_i x_i e^{\beta x_i}}{\sum_i e^{\beta x_i}}.
\]

实现细节：

- 先对每个接收者、每个维度取最大值 \(x_{\max}\)，计算
  \(\exp(\beta(x_i-x_{\max}))\)，避免指数溢出；
- 最大值平移只用于数值稳定，梯度不回传到平移量；
- 没有收到消息的对象输出零向量；
- 聚合对消息顺序不变。

### 6.4 残差更新

所有对象共享同一个更新网络：

\[
\operatorname{Comb}_U:\mathbb R^{2d}\rightarrow\mathbb R^{d},
\]

\[
X_{i+1}(o)=X_i(o)+\operatorname{Comb}_U\bigl(X_i(o)\Vert M(o)\bigr).
\]

执行 \(L=4\) 层。`Comb_U` 在所有 Cell/Action 对象之间共享，`Comb_p` 不共享。

### 6.5 全局池化

完成 \(L\) 层后，对 Cell Objects 与 Action Objects 一起做置换不变的 SmoothMax 池化：

\[
\bar X(s,g)=\operatorname{Pool}\bigl(\{X_L(o):o\in\widetilde O\}\bigr)\in\mathbb R^{32}.
\]

### 6.6 共享 Q Readout

对每个动作对象 \(o_a\)：

\[
Q_\theta(s,a)=\operatorname{MLP}_Q\bigl(X_L(o_a)\Vert \bar X(s,g)\bigr),
\]

输入 64 维，隐藏层 64，输出 1 个标量。MLP_Q 在所有动作对象之间共享，UP/DOWN/LEFT/RIGHT 不建立独立 Q 网络。

只有当某个动作出现在图中（即合法）时，它才有对应的 Q 值；非法动作不参与搜索，也不参与目标网络的 \(\max\)。

### 6.7 设备与批处理检查

- 网络参数、关系 Tensor、动作对象索引和 Q 输入必须位于同一 CUDA 设备；
- 训练 Batch 采用真实 disjoint-union 图批处理，不使用“固定 256 格 + 固定 4 动作”的错误填充；
- 不同地图的图批处理不会产生跨图边。

---

## 7. GSP 状态—动作 WA*

### 7.1 优先级

搜索节点是 \((s,a)\)。优先级最大化：

\[
f(s,a)=g(s)+wQ_\theta(s,a),
\qquad
g(s)=-C(start,s),\quad w=2.
\]

`g(s)` 是从起点到当前状态的累计地形成本取负值，因此 \(g(s)\le 0\)。使用 Python `heapq` 加负优先级实现最大堆，并以单调计数器处理同分排序，保证结果确定。

### 7.2 搜索预算

所有受预算约束的算法统一使用：

\[
B=512
\]

**状态转移评估次数（transition evaluations）**。

GSP 中，每检查/生成并评价一个合法状态—动作候选，计数一次。对某个状态，若其全部合法动作无法在剩余预算内完整评估，则搜索停止并记录 `transition_budget`，不会只评估一部分动作。传统 A*/WA* 同样按合法后继转移计数。

### 7.3 搜索流程

1. 初始化起点状态与累计成本 `cost[start]=0`；
2. 构建起点的关系图，R-GNN 计算所有合法动作 Q 值；
3. 对每个合法动作计算 `f(start,a)`，加入 Frontier；
4. 弹出当前 `f` 最大的状态—动作对；
5. 执行状态转移，得到 `next_position`、实际地形成本和奖励；
6. 记录父节点、动作、奖励、累计成本和 Frontier 统计；
7. 若到达目标，立即停止并回溯成功路径；
8. 若后继没有合法动作且不是目标，记录真正死路并跳过；
9. 若后继是首次发现，则设置父节点和累计成本，构建其关系图并评估所有合法动作；
10. 重复直至找到目标、预算耗尽或 Frontier 为空。

搜索采用**首次发现父链**：每个状态只扩展一次，不重新扩展，不做父节点重连。这是本项目的搜索树策略，使 `g(s)` 固定为该状态首次被发现时的累计成本。

### 7.4 推理缓存与冻结

- 搜索前把网络切到 `eval()`，临时关闭参数 `requires_grad`，并在 `torch.inference_mode()` 下推理；
- 每个 `(地图, 当前位置, 目标)` 的 Q 值在一次搜索内缓存，同一状态只前向一次；
- 静态地图关系和动态状态图通过 `RelationalGraphBuilder` 缓存；
- 搜索不进行任何梯度更新。

### 7.5 搜索统计与路径回报

每次搜索记录：

- 父节点、状态、动作、实际移动成本、累计路径成本；
- 状态—动作扩展数 `node_expansions`；
- 不同状态扩展数 `unique_state_expansions` 与重复扩展数 `reexpansions`；
- 转移评估次数 `transition_evaluations`；
- Frontier 大小轨迹与最大 Frontier；
- 终止原因（`goal` / `transition_budget` / `frontier_exhausted`）；
- GNN 推理耗时与 Q 评估次数；
- 所有产生的 TransitionExperience。

找到目标后沿父链回溯：

\[
R_t=\sum_{j=t}^{T-1} r_j=-\sum_{j=t}^{T-1} c_j,
\]

即每个成功路径边的 `return_bound` 被替换为**从该边之后到目标的实际剩余回报**。普通非成功路径转移的 `return_bound=-inf`。

### 7.6 真正死路

如果某个后继不是目标且没有任何合法动作，记录为真正死路，回报下界：

\[
R_\perp=-2048.
\]

本环境是连通的无向四方向网格，任何非目标可达状态都至少有回到前驱的合法动作，因此实际搜索树中通常不存在真正的 `A(s)=∅` 死路。程序保留该分支并如实计数，**不会把视觉死胡同人为标记为 dead-end**。16×16 简单路径最多 255 条边、每条边最大成本 5，最坏可行回报不低于 `-1275`，所以 `-2048` 是严格低于任何可行路径回报的下界。

---

## 8. Replay Buffer

[`replay_buffer.py`](replay_buffer.py) 保存紧凑 Transition，不保存 192/768 维固定 MLP 输入：

```python
{
    "instance_id": ...,     # 引用地形矩阵与静态图
    "position": ...,        # (row, col)
    "goal": ...,            # (row, col)
    "action": ...,          # 0..3
    "next_position": ...,   # (row, col)
    "reward": ...,          # -1 / -3 / -5
    "return_bound": ...,    # -inf / 有限成功路径回报 / -2048
    "terminal": ...,        # 是否到达目标
    "dead_end": ...,        # 是否真正无合法动作
}
```

- FIFO 容量默认 20000；
- 所有迷宫、所有 Episode 共用一个 Buffer；
- 训练时随机均匀抽取 64 条；
- 采样后通过 `instance_id` 取回地形，再构建/缓存当前状态与后继状态的关系图；
- 不把 Dijkstra 成本或最优路径写入 Buffer。

---

## 9. Q-learning 训练目标

### 9.1 Online 与 Target Network

Online \(Q_\theta\) 和 Target \(Q_{\theta^-}\) 采用完全相同的 R-GNN 结构。初始化时 Target 复制 Online；Target 不参与梯度，每 **200 次梯度更新**硬同步一次。

### 9.2 Bellman 估计

\[
\hat y(s,a)=r(s,a,s')+\gamma\max_{a'\in A(s')}Q_{\theta^-}(s',a'),
\qquad \gamma=1.
\]

后继状态的所有合法动作由图中的 Action Objects 给出；`action_max_per_graph` 按图取最大值，没有合法动作的图返回 `-inf`。非终止、非死路样本若没有合法后继，程序会报错，而不是静默使用非法动作。

### 9.3 最终目标

| 样本类型 | 目标 \(y\) |
|---|---|
| 普通非终止转移 | \(\hat y(s,a)\) |
| 成功路径上的非终止转移 | \(\max(R,\hat y(s,a))\) |
| 直接到达目标 | \(r(s,a,goal)\)，不 Bootstrap |
| 真正死路 | \(R_\perp=-2048\) |

`R` 是搜索实际找到路径的回报下界。消融 `no_solution_bounds` 会关闭有限 \(R\) 的使用，使成功路径非终止样本退回普通 Bellman target；终止与死路目标保持不变。

### 9.4 损失与优化器

\[
L(\theta)=\frac1N\sum_{i=1}^{N}\bigl(Q_\theta(s_i,a_i)-y_i\bigr)^2.
\]

| 项目 | 设置 |
|---|---|
| Loss | MSE |
| Optimizer | Adam |
| R-GNN 主体学习率 | `1e-4` |
| Q Readout 学习率 | `1e-3` |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Dropout / BatchNorm | 不使用 |
| Batch size | 64 |
| Replay capacity | 20000 |
| 每 Episode 最多梯度更新 | 2 |

一次更新流程：采样 Batch → 构建当前状态图 → Online 前向 → 按动作索引取实际动作 Q → Target 前向后继图 → 合法动作取最大 → 构造目标 → MSE → 反向传播 → 梯度裁剪 → Adam 更新 → 按周期同步 Target。

训练全程不使用 Dijkstra 最优路径作为监督标签。

---

## 10. 训练闭环与动态实例采样

**一个 Episode = 选择一张训练迷宫 + 执行一次完整 GSP-WA* + 将全部经验写入 Replay Buffer + 最多两次梯度更新。**

每个 Episode：

1. 按实例池动态采样选择一个训练迷宫；
2. 使用当前 Online Network 执行 512 预算状态—动作 WA*；
3. 将本次搜索的全部 Transition 写入共享 Buffer；
4. 若 Buffer 少于 64 条，不做梯度更新；
5. 否则独立抽取两个 Batch，各做一次 Q-learning 更新；
6. 更新实例池：
   - `unsolved`：搜索失败；
   - `solved`：成功，且 `node_expansions == plan_length`；
   - `satisficed`：成功，但 `node_expansions > plan_length`；
7. 按非空池权重 `unsolved:solved:satisficed = 1:2:4` 做归一化采样。

一个实例只存在于最新结果对应的池中。权重是项目工程初始值，不宣称是论文原始数值。

验证：

- 默认每 100 Episode 在固定验证集上评估；
- 使用与训练相同的 512 转移预算；
- 不更新参数、不写 Replay Buffer、不修改实例池；
- 模型选择顺序：验证成功率更高 → 成功实例平均 PCR 更低 → 平均转移评估数更低；
- 最后一个 Episode 也会验证并保存 best/last。

Checkpoint 与日志：

```text
checkpoints/16x16_rgnn/
├── initial_model.pt        # 训练开始前的随机初始网络
├── best_model.pt           # 验证最优
├── last_model.pt           # 最后一个验证点
├── config.json             # 实际运行配置
└── training_summary.json   # 训练摘要

logs/16x16_rgnn/
├── training.log
├── training_log.csv        # 每个 Episode 汇总
├── validation_log.csv      # 每个验证点
└── update_log.csv          # 每次梯度更新
```

`initial_model.pt` 在任何训练或 Target 构造之前保存，用于 Untrained R-GNN 对照。

---

## 11. 传统算法与最终评价

### 11.1 五种固定算法

| 算法 | 优先级/作用 | 预算 |
|---|---|---|
| Dijkstra | 最小累计地形成本，最优参考 | 不受 512 限制 |
| A* + Manhattan | \(f(s)=g_{\text{cost}}(s)+h(s)\)，\(h=d_{\text{Manhattan}}\times c_{\min}\)，\(c_{\min}=1\) | 512 |
| WA* + Manhattan | \(f(s)=g_{\text{cost}}(s)+2h(s)\) | 512 |
| Untrained R-GNN + WA* | 随机初始 R-GNN，\(f(s,a)=g(s)+2Q_\theta(s,a)\)，不更新参数 | 512 |
| Trained GSP-RGNN + WA* | 训练后 R-GNN，同一 WA* 框架 | 512 |

所有算法使用相同的测试图、起点、终点和地形矩阵。Dijkstra 在每次评估时重新计算当前 split 全部测试图的最优成本，并与数据集保存的 `optimal_cost` 校验一致。

### 11.2 固定评价指标（不再增加）

1. **成功率**

\[
SR=\frac{N_{\text{success}}}{N_{\text{test}}}\times 100\%.
\]

2. **加权路径成本比 PCR**

\[
PCR_i=\frac{C_i}{C_i^*}.
\]

`C_i` 是算法实际找到的累计地形成本，`C_i*` 是 Dijkstra 最优成本。只对成功实例计算均值，失败实例不会被伪造路径成本。同时记录成功实例平均绝对路径成本。

3. **不同状态扩展数 Unique State Expansions**

同一状态重复访问时单独记录 `reexpansions`，不同状态只计一次。GSP 的首次发现策略下重复扩展通常为 0。

4. **状态转移评估次数 Transition Evaluations**

每次检查、生成并评价一个合法状态转移计数一次。GSP 每个合法状态—动作候选计数一次；A*/WA* 每个合法后继转移计数一次；重复生成或重新评价按实际发生次数计数。这是跨算法比较搜索工作量的主要指标。

5. **端到端搜索时间 Search Time**

单位 ms，包含图构建、R-GNN 推理、Frontier 维护、状态转移和路径回溯；不包括模型加载和基准测试前的预热。CUDA 计时在前后同步。GNN 推理时间单独记录，只作为 Search Time 的组成部分分析，不作为独立排名指标。

按五种结构分别统计上述指标：`random_obstacle`、`rooms_corridors`、`u_detour`、`branching_maze`、`loop_maze`。

### 11.3 输出

```text
results/16x16_rgnn/
├── evaluation.csv          # 每算法 × 每测试图一行
├── structure_analysis.csv  # 每算法 × 每结构汇总
├── summary.json            # 算法总体、结构分组、配置与定义
├── evaluation.log
└── figures/
```

`evaluation.csv` 列：

```text
algorithm, instance_id, structure, success, path_cost, optimal_cost, pcr,
unique_state_expansions, reexpansions, transition_evaluations,
search_time_ms, gnn_inference_time_ms, failure_reason, budget_limited
```

---

## 12. 项目结构

```text
GSP-Reproduction-V2/
├── config.py                 # 所有环境、R-GNN、搜索、训练、评估参数
├── environment.py            # 加权迷宫、合法性、BFS、Dijkstra、地形统计
├── generator.py              # 五种结构、地形分配、数据集生成与加载
├── relational_graph.py       # 对象/谓词/动作对象、静态缓存、图批处理
├── model.py                  # 谓词特定 R-GNN、SmoothMax、共享 Q Readout
├── search.py                 # GSP 状态—动作 WA*、经验与回报下界
├── replay_buffer.py          # 紧凑 Transition FIFO 回放
├── trainer.py                # Q-learning、Target Network、验证、实例池
├── train.py                  # 独立训练入口
├── baselines.py              # Dijkstra、A*、WA*
├── evaluate.py               # 独立五算法评估入口
├── metrics.py                # 固定指标与 CSV/JSON
├── visualize.py              # 从已有 CSV/JSON 生成图表
├── utils.py                  # 种子、日志、checkpoint、计时、迷宫绘图
├── check_cuda.py             # 严格 CUDA 检查
├── requirements.txt
├── README.md
├── data/16x16_rgnn/          # 正式数据（待授权生成）
├── checkpoints/16x16_rgnn/   # 正式模型（待授权训练）
├── logs/16x16_rgnn/
├── results/16x16_rgnn/
│   └── figures/
└── tests/
    ├── test_environment.py
    ├── test_generator.py
    ├── test_relational_graph.py
    ├── test_model.py
    ├── test_search.py
    └── test_training.py
```

本项目**不包含** `run_experiments.py`；每个阶段由独立脚本执行。

---

## 13. 环境与运行命令

### 13.1 本机验证环境

```text
Python:       3.13.5 (Anaconda, conda base)
PyTorch:      2.11.0+cu128
CUDA build:   12.8
GPU:          NVIDIA GeForce RTX 5060 Laptop GPU
Compute:      12.0
GPU memory:   7.96 GiB
```

`requirements.txt` 要求 `torch>=2.0`、`numpy>=1.24`、`matplotlib>=3.7`、`tqdm>=4.66`、`pytest>=8.0`。CUDA 不可用时训练和评估会直接报错，不会静默回退 CPU。

### 13.2 检查 CUDA

```bash
python check_cuda.py
```

### 13.3 生成正式数据集

```bash
python generator.py --seed 42
```

默认生成 1000/200/300 张。若 `data/16x16_rgnn/` 已有文件，默认拒绝覆盖；必须显式加 `--force`。

冒烟/调试可生成小数据集：

```bash
python generator.py --seed 42 \
    --data-dir .tmp_smoke/data \
    --log-dir .tmp_smoke/logs \
    --train-size 100 --validation-size 20 --test-size 20
```

三个 size 必须都能被 5 整除。

### 13.4 运行测试

```bash
pytest -q
```

### 13.5 正式训练（尚未启动）

```bash
python train.py \
    --device cuda:0 \
    --episodes 3000 \
    --seed 42
```

可用参数：`--data-dir`、`--checkpoint-dir`、`--log-dir`、`--validation-interval`、`--batch-size`、`--search-budget`、`--updates-per-episode`。

消融（单独调用 `train.py`）：

```bash
# 关闭成功路径搜索回报下界
python train.py --device cuda:0 --seed 42 --ablation no_solution_bounds

# 关闭动态实例池，改为均匀抽样
python train.py --device cuda:0 --seed 42 --ablation uniform_sampling
```

消融输出自动进入 `checkpoints/16x16_rgnn/ablations/<name>/` 和 `logs/16x16_rgnn/ablations/<name>/`。本环境没有真实死路样本，因此不提供无意义的 Dead-end 消融。

### 13.6 模型评估

```bash
python evaluate.py \
    --checkpoint checkpoints/16x16_rgnn/best_model.pt \
    --initial-checkpoint checkpoints/16x16_rgnn/initial_model.pt \
    --device cuda:0 \
    --split test
```

可单独运行某一算法：

```bash
python evaluate.py --checkpoint ... --algorithms dijkstra
python evaluate.py --checkpoint ... --algorithms astar wastar
python evaluate.py --checkpoint ... --algorithms untrained_rgnn trained_gsp
```

`--output-dir` 可指定独立结果目录，避免覆盖正式结果。

### 13.7 绘图

```bash
python visualize.py \
    --log-dir logs/16x16_rgnn \
    --result-dir results/16x16_rgnn
```

只读取已有 CSV/JSON，不重新运行搜索或训练。生成 4 条训练曲线和 5 项测试指标图，共 9 张 PNG：

```text
training_loss.png
validation_success_rate.png
validation_weighted_path_cost_ratio.png
validation_transition_evaluations.png
test_success_rate.png
test_weighted_path_cost_ratio.png
test_unique_state_expansions.png
test_transition_evaluations.png
test_search_time.png
```

---

## 14. 本次验证结果

> 本节包括单元测试、正式数据集生成、冒烟训练/评估与图表检查。冒烟训练只有 12 Episode，数据只有 100/20/20，**不能作为正式实验结论**。正式 3000 Episode 训练尚未启动。

### 14.1 单元测试

```text
70 passed in 11.11s
```

| 测试文件 | 用例数 | 覆盖 |
|---|---:|---|
| `test_environment.py` | 16 | 地形编码、墙壁、奖励、BFS、Dijkstra、路径成本、哈希 |
| `test_generator.py` | 16 | 五结构、均衡计数、连通性、地形随机性、数据集独立性、U 型装饰墙不覆盖端点 |
| `test_relational_graph.py` | 9 | 256 Cell Objects、动作对象、谓词元数、批处理隔离、重映射 |
| `test_model.py` | 9 | 独立 Comb_p、角色消息、SmoothMax、残差、池化、共享 Readout、置换一致性 |
| `test_search.py` | 9 | 状态—动作优先级、预算、回报下界、累计成本、Dijkstra/A*/WA* |
| `test_training.py` | 11 | Bellman Target、终止不 Bootstrap、合法动作 max、Replay、Target 同步、CUDA 反传、两种消融、指标 |

所有测试均为实际 PASS/FAIL，没有跳过关键测试。

### 14.2 CUDA 检查

```text
CUDA available: True
NVIDIA GPU name: NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch CUDA version: 12.8
PyTorch version: 2.11.0+cu128
Selected device index: 0 (cuda:0)
Compute capability: 12.0
GPU memory total: 7.96 GiB
```

训练日志确认：

```text
Online parameter device: cuda:0
Target parameter device: cuda:0
First graph batch device audit:
  online_parameters=cuda:0, target_parameters=cuda:0,
  graph_relations=cuda:0, q_input_objects=cuda:0
CUDA reverse pass verified
```

### 14.3 冒烟数据集

```text
.tmp_smoke/data/
├── train.npz            100 张（每结构 20）
├── validation.npz        20 张（每结构 4）
├── test.npz              20 张（每结构 4）
└── dataset_manifest.json
```

三个 split 的地形布局哈希无重叠；五结构数量精确均衡；地形按 1/3/5 独立随机分配。

### 14.4 正式数据集生成（已完成）

首次生成正式数据时，`u_detour` 的附加随机墙块可能覆盖终点，导致
`bfs_shortest_path` 抛出 `ValueError: Start and goal must be walkable cells`。
修复方式：装饰墙块跳过起点/终点格，并在 BFS 前再次检查端点可通行；
同时加入回归测试 `test_u_detour_decorative_walls_never_block_endpoints`。

修复后执行：

```bash
python generator.py --seed 42
```

实际结果（约 12 秒）：

```text
data/16x16_rgnn/
├── train.npz            1000 张（每结构 200）
├── validation.npz        200 张（每结构 40）
├── test.npz              300 张（每结构 60）
└── dataset_manifest.json
```

独立性审计：

```text
train__validation: 0
train__test:       0
validation__test:  0
within-split duplicates: 0
unique maps: 1000 / 200 / 300
```

三档地形总体比例约为 `33.3% / 33.5% / 33.2%`，与结构类型无关。
正式训练所需的数据已就绪。

### 14.5 12 Episode 冒烟训练

命令：

```bash
python train.py \
    --device cuda:0 \
    --episodes 12 \
    --seed 42 \
    --data-dir .tmp_smoke/data \
    --checkpoint-dir checkpoints/smoke_12ep_seed42 \
    --log-dir logs/smoke_12ep_seed42 \
    --validation-interval 4
```

结果：

- 训练时间：3 分 49 秒（含 3 次验证）；
- 梯度更新：24 次；
- Target Network 同步：0 次（尚未达到 200 次更新阈值；同步逻辑由单元测试覆盖）；
- Replay Buffer 最终大小：1851；
- 最后一个 Episode 的聚合损失：5.4291；
- GPU 峰值显存：610.43 MiB；
- 验证（20 张）：成功率 75.00%（15/20），成功实例 PCR 1.0000，平均不同状态扩展 114.15，平均转移评估 328.45；
- best checkpoint：Episode 4。

产物：

```text
checkpoints/smoke_12ep_seed42/{initial_model.pt,best_model.pt,last_model.pt,config.json,training_summary.json}
logs/smoke_12ep_seed42/{training.log,training_log.csv,validation_log.csv,update_log.csv}
```

### 14.6 冒烟评估

命令：

```bash
python evaluate.py \
    --checkpoint checkpoints/smoke_12ep_seed42/best_model.pt \
    --initial-checkpoint checkpoints/smoke_12ep_seed42/initial_model.pt \
    --device cuda:0 \
    --split test \
    --data-dir .tmp_smoke/data \
    --output-dir results/smoke_12ep_seed42
```

20 张测试图 × 5 算法 = 100 次运行，完整流程 1 分 34 秒。

| 算法 | 成功率 | PCR | 平均不同状态扩展 | 平均转移评估 | 平均搜索时间 |
|---|---:|---:|---:|---:|---:|
| Dijkstra | 100.00% | 1.0000 | 107.95 | 320.45 | 1.13 ms |
| A* + Manhattan | 80.00% | 1.0000 | 82.45 | 239.30 | 1.41 ms |
| WA* + Manhattan | 80.00% | 1.0253 | 73.35 | 215.70 | 1.24 ms |
| Untrained R-GNN + WA* | 75.00% | 1.0000 | 95.50 | 276.20 | 2335.98 ms |
| Trained GSP-RGNN + WA* | 75.00% | 1.0000 | 94.95 | 274.35 | 2365.96 ms |

R-GNN 两种算法的平均 GNN 推理时间分别约 2078 ms 和 2102 ms，占搜索时间的大部分。20 张测试图、12 Episode 训练不足以体现学习收益；Trained 与 Untrained 的差异在此规模下没有统计意义。

产物：

```text
results/smoke_12ep_seed42/
├── evaluation.csv
├── structure_analysis.csv
├── summary.json
├── evaluation.log
└── figures/            # 9 张 PNG
```

### 14.7 图表检查

`visualize.py` 实际生成 9 张图，训练曲线与测试算法对比图均可正常打开：

```text
results/smoke_12ep_seed42/figures/
├── training_loss.png
├── validation_success_rate.png
├── validation_weighted_path_cost_ratio.png
├── validation_transition_evaluations.png
├── test_success_rate.png
├── test_weighted_path_cost_ratio.png
├── test_unique_state_expansions.png
├── test_transition_evaluations.png
└── test_search_time.png
```

### 14.8 尚未执行

- 正式数据集已生成；未启动 3000 Episode 正式训练；
- 未生成正式 `checkpoints/16x16_rgnn/`、`logs/16x16_rgnn/`、`results/16x16_rgnn/`；
- 未做三随机种子正式实验。

按任务书要求，正式训练等待用户显式授权后再执行。

---

## 15. 如何解读正式结果

建议按以下顺序分析：

1. **成功率**：先看 Trained GSP-RGNN + WA* 相对 Untrained R-GNN + WA* 是否提高。A*/WA* 在 16×16 加权图上受 512 转移预算限制，不一定达到 100%；Dijkstra 不受预算限制，应始终为 100%。
2. **PCR**：只对成功实例计算，越接近 1 越好。必须与成功率一起报告；失败实例没有被分配路径成本。
3. **不同状态扩展数 / 转移评估数**：这是搜索工作量的主指标。GSP 按状态—动作对评估，传统算法按状态扩展，不能把两边的“扩展数”直接等同。
4. **搜索时间**：R-GNN 需要为每个新状态执行一次图前向，通常显著慢于传统基线。应结合 `gnn_inference_time_ms` 分析，但这只是 Search Time 的组成部分，不作为独立排名指标。
5. **结构分组**：五种结构的成功率、PCR、扩展数和时间都要分别查看。某一结构上的退化不能被总体均值掩盖。
6. **随机种子**：正式结论应基于三个种子的均值和样本标准差；单种子差异不能直接归因于算法。
7. **训练曲线**：只看四条固定曲线（Training Loss、Validation Success Rate、Validation Weighted PCR、Validation Transition Evaluations），不要从冒烟结果推断最终性能。

预期中的合理现象：

- Dijkstra 成功率 100%，是成本参考，不参与预算排名；
- A*/WA* 在 512 转移评估预算下可能因复杂地形无法到达目标，尤其 `rooms_corridors`、`u_detour`、`branching_maze`；
- 学到的 R-GNN 在训练早期可能不如 Manhattan WA*，训练充分后才可能缩小差距；
- 由于 R-GNN 推理在图构建之后逐状态进行，搜索时间明显高于纯 CPU 传统搜索是正常的。

---

## 16. 可复现性与限制

- 每个 split 使用独立 master seed，每张地图保存自己的生成种子；
- Train/Validation/Test 按地形布局 SHA-256 去重，正式数据生成会审计跨集合重叠并拒绝非零重叠；
- Python、NumPy、PyTorch RNG 统一设置，并启用确定性配置；
- 实际运行配置保存到 checkpoint 目录的 `config.json`；
- 训练、验证、评估均不读取测试集标签；Dijkstra 参考成本只用于评估；
- 禁止在测试集上早停、调参或模型选择；
- 搜索首次发现父链不重连、不重复扩展，这是本项目的搜索树策略；
- 真正死路在连通无向网格中通常不存在，死路样本数可能为 0，这是如实结果；
- 本项目不声称与论文原始代码一致，也不声称跨尺寸或跨关系结构泛化；
- 逐谓词 Comb_p 的批处理是算子级优化，不改变参数共享和消息定义；
- 正式训练的计算量较大：在当前 RTX 5060 Laptop GPU 上，冒烟流程约 2.3–6.6 秒/Episode（含图构建、推理和更新），3000 Episode 加周期验证预计需要数小时，实际时间取决于地图复杂度和 CUDA 状态。

---

## 17. 任务书验收对照

| 验收项 | 实现位置 | 验证方式 |
|---|---|---|
| 16×16 地图生成 | `environment.py`、`generator.py` | `test_environment.py`、`test_generator.py` |
| 五结构均衡 | `generator.py` | 冒烟数据 manifest、`test_generator.py` |
| 地形 1/3/5 与奖励 -1/-3/-5 | `environment.py` | `test_environment.py` |
| 墙壁不可通行 | `environment.py`、`relational_graph.py` | `test_environment.py`、`test_relational_graph.py` |
| BFS 连通性与 Dijkstra 最优成本 | `environment.py`、`generator.py` | `test_environment.py`、`test_generator.py` |
| 数据集无重复 | `generator.py` | `audit_splits`、`test_generator.py` |
| 256 Cell Objects | `relational_graph.py` | `test_relational_graph.py` |
| 动态 Action Objects | `relational_graph.py` | `test_relational_graph.py` |
| 14 种谓词与角色消息 | `relational_graph.py`、`model.py` | `test_relational_graph.py`、`test_model.py` |
| 独立 Comb_p | `model.py` | `test_model.py` |
| SmoothMax 稳定性与置换不变性 | `model.py` | `test_model.py` |
| Residual Update | `model.py` | `test_model.py` |
| Global Pooling | `model.py` | `test_model.py` |
| 共享 Q Readout | `model.py` | `test_model.py` |
| 变长动作数量前向 | `relational_graph.py`、`model.py` | `test_model.py` |
| 对象重编号一致性 | `relational_graph.py`、`model.py` | `test_model.py` |
| 多图 disjoint-union 隔离 | `relational_graph.py` | `test_relational_graph.py` |
| 状态—动作 WA* | `search.py` | `test_search.py` |
| 累计地形成本 | `environment.py`、`search.py` | `test_search.py` |
| 成功路径回报回溯 | `search.py` | `test_search.py` |
| Bellman Target 与合法动作 max | `trainer.py` | `test_training.py` |
| 终止不 Bootstrap | `trainer.py` | `test_training.py` |
| Replay 保存与恢复 | `replay_buffer.py` | `test_training.py` |
| Online/Target 正常工作 | `trainer.py` | `test_training.py` |
| CUDA 前向/反向 | `check_cuda.py`、`trainer.py` | 冒烟训练日志、`test_training.py` |
| 模型保存/加载一致 | `utils.py` | `test_model.py`、`test_training.py` |
| 固定评估指标与结构分组 | `metrics.py`、`evaluate.py` | `test_training.py`、冒烟评估 CSV/JSON |
| 独立命令行入口 | 各脚本 `main()`/argparse | `--help`、本次冒烟命令 |
| 冒烟训练 10–20 Episode | `train.py` | 本次 12 Episode |
| 冒烟 checkpoint 评估 | `evaluate.py` | 本次 20 张测试图 × 5 算法 |
| 图表生成 | `visualize.py` | 本次 9 张 PNG |

---

## 18. 结论与下一步

V2 已完成从加权网格到关系事实、从 Cell/Action Objects 到谓词消息传递、从共享 Q Readout 到 GSP-WA*、从搜索经验到 Q-learning 的完整闭环，并在 CUDA 上通过了 70 项单元测试和一次 12 Episode 冒烟训练、五算法评估与绘图验证。正式数据集 1000/200/300 已生成并通过独立性审计。

**正式训练尚未开始。** 需要用户授权后依次执行：

```bash
# 数据集已生成；如需强制重建：python generator.py --seed 42 --force
python train.py --device cuda:0 --episodes 3000 --seed 42
python evaluate.py --checkpoint checkpoints/16x16_rgnn/best_model.pt \
    --initial-checkpoint checkpoints/16x16_rgnn/initial_model.pt \
    --device cuda:0 --split test
python visualize.py
```

正式结果将写入 `checkpoints/16x16_rgnn/`、`logs/16x16_rgnn/` 和 `results/16x16_rgnn/`。