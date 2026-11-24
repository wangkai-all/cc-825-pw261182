# 多期情景缩减与尾部保真工作台

一个**仅依赖 Python 3.11 标准库和原生 HTML/CSS/JavaScript** 的端到端情景数据工具。它不是聚类散点图页面，而是把以下环节串成可运行、可检查、可恢复的流程：

1. 导入虚构的多期路径与概率；
2. 先合并完全重复路径的概率，再分别检查零概率、负概率和未归一化；
3. 联合优化“选哪些代表路径”和“代表概率如何再分配”；
4. 按业务量纲、维度权重、时间先后和共同前缀定义距离；
5. 用前缀动态规划保持非预知结构，并构建/校验情景树；
6. 用尾部权重、高损失非对称代价和硬尾部守卫保护少见但代价大的情景；
7. 同时显示平均分布距离与极端损失保真的代价，输出 Pareto/权衡表；
8. 把原始情景集和缩减情景集送入同一个小型多期规划模型，并在完整原始分布上评估缩减决策的样本外遗憾；
9. 对 8 路径小实例枚举全部 `C(8,3)=56` 个子集作为最优性对照；
10. 提供崩溃—恢复演练，证明已提交的概率质量不会因重复消息再次分配。

---

## 1. 快速开始

要求 Python 3.11+，无需安装第三方包。

```bash
cd /workspace
python3 -m scenario_tool.server
```

打开：

```text
http://localhost:8000
```

也可以指定端口：

```bash
PORT=8123 python3 -m scenario_tool.server
# 或
python3 -m scenario_tool.cli serve --host 127.0.0.1 --port 8123
```

或使用启动脚本：

```bash
./run.sh
```

页面打开后会自动尝试载入示例；也可以点击：

- **180 条示例**：带正常、高需求、风暴、严重中断分支的虚构数据；
- **生成 10,000 条**：压测规模路径；
- **小实例**：8 条带明确共同前缀和严重尾部 `C2` 的路径；
- **导入 JSON/CSV**：导入自己的虚构数据。

---

## 2. 测试与命令行

运行完整 `unittest` 测试集：

```bash
python3 -m unittest discover -s tests -v
```

检查前端脚本语法：

```bash
node --check web/app.js
```

CLI：

```bash
# 检查、合并重复路径并输出结构化报告
python3 -m scenario_tool.cli validate examples/small_prefix_tree.json

# 缩减并比较规划模型；默认向标准输出打印摘要
python3 -m scenario_tool.cli reduce examples/fictional_paths_120.json \
  --k 12 --tail-weight 1.5 --tail-level 0.95 --tail-guards 2

# 输出完整 JSON 结果
python3 -m scenario_tool.cli reduce examples/fictional_paths_120.json \
  --k 10 --out /tmp/reduction-result.json

# 枚举小实例全部代表子集
python3 -m scenario_tool.cli exact-check

# 崩溃恢复幂等演练
python3 -m scenario_tool.cli recovery-drill
```

如果处理的是下载或解压得到的不受信任数据，应把数据放在独立空目录，脚本保留在项目目录，并从项目目录把数据路径作为参数传入；必要时使用 Python isolated mode：

```bash
python3 -I -m scenario_tool.cli validate /absolute/path/to/untrusted-input.json
```

---

## 3. 网页工作流

### 3.1 导入检查

导入后首先执行规范化和重复路径合并。重复判定基于完全相同的规范化路径值，而不是 ID 或标签；重复行的概率先求和，然后才执行概率检查。

检查结果明确区分：

- `DUPLICATE_PATHS_MERGED`：合并了多少重复行、多少概率质量；
- `ZERO_PROBABILITY`：存在精确零概率路径；
- `NEGATIVE_PROBABILITY`：存在非法负概率；
- `UNNORMALIZED_PROBABILITY`：概率和与 1 的偏差超过容差；
- `SCHEMA_*`：期间、维度、路径矩阵形状或数值类型问题。

零概率路径不会被静默删除：它们仍可在浏览器中查看和分配，但不能成为正质量代表。未归一化输入不会被偷偷修改；只有点击 **“归一化概率后继续”** 或在 API/CLI 中显式设置 `normalize=true` 后才归一化。

### 3.2 缩减与尾部保护

可调参数包括：

- `K`：代表路径数；
- `tailWeight`：极端损失保真在联合目标中的权重；
- `tailLevel`：90%、95% 或 99% 尾部水平；
- `tailGuards`：强制保留的最高损失路径数量；
- `distanceMetric`：
  - `prefix`：前缀累积距离，前期分歧会在后续所有前缀中持续计费；
  - `chronological`：普通分期加权距离；
- `timeCurve`：`early`、`uniform`、`late`；
- `prefixLocked`：是否强制非预知前缀映射。

页面 KPI 会显示：

- 代表数、原始路径数；
- 概率守恒误差；
- 平均业务距离；
- CVaR 误差和尾部低估；
- 规划决策 `Q` 的变化；
- 样本外规划遗憾；
- 前缀兼容性和树校验状态。

### 3.3 两种代价必须同时可见

点击 **“生成平均距离/尾部损失代价前沿”** 后，系统会对多个尾部权重分别独立运行联合优化。

每个点都记录：

- 平均概率加权业务距离；
- 相对距离优先解的“尾部保护距离加价”；
- CVaR 绝对误差；
- CVaR 低估量；
- 最坏损失偏差；
- 规划遗憾；
- 缩减规划决策 `Q`；
- 代表 ID 列表。

因此可以直接看到：把罕见严重路径保护得更好时，平均距离通常会上升；只最小化平均距离时，可能得到较小的分布距离，却低估 CVaR 或选出样本外表现更差的库存决策。

### 3.4 原路径 → 代表浏览器

第 4 区可以：

- 点击左侧代表，查看该代表吸收了多少原始概率质量；
- 搜索原始路径 ID 或标签；
- 查看每条原路径分配给了哪个代表；
- 查看单路径距离、规划损失差和是否属于尾部；
- 切换维度查看原路径与代表路径的逐期曲线。

### 3.5 非预知情景树

共同前缀不是只用于画图。前缀兼容分配满足：

> 如果若干原始路径在深度 `d` 前拥有相同前缀，那么它们所映射的代表路径也必须在缩减树中共享深度 `d` 前缀。

这防止把同一个当前信息节点下的路径映射到提前分叉的代表，从而“提前知道未来”。

后端使用前缀树 DP 求给定代表集合下的最优前缀兼容分配，并另有独立检查器报告任何违规。前端第 5 区渲染可展开的前缀树，节点概率、条件概率和前缀值均可见。

### 3.6 同一个小规划模型

缩减前后使用同一个多期库存/应急采购模型：

- 第一阶段决定初始库存/采购量 `Q`；
- 每期先消耗现有库存满足需求；
- 缺货通过应急采购补足；
- 应急单位成本受各期 `freight` 和 `disruption` 影响；
- 剩余库存产生持有成本，期末剩余有残值。

比较分三层：

1. 在完整原始分布上求解决策 `Q_original`；
2. 在缩减分布上求解决策 `Q_reduced`；
3. 固定 `Q_reduced`，回到完整原始分布评估真实期望成本，并计算：

```text
out_of_sample_regret =
    E_original[Cost(Q_reduced)]
  - E_original[Cost(Q_original)]
```

遗憾值按构造应非负（数值噪声范围内）。

### 3.7 完全搜索对照

内置 8 路径实例在 `K=3` 时枚举全部 56 个代表子集，并对每个子集执行相同的联合目标、概率再分配、前缀检查和树构建。

页面第 7 区显示：

- 枚举子集数；
- 完全搜索最优值和代表；
- 自动/启发式方法的目标值和代表；
- 目标差距；
- 是否匹配完全搜索最优；
- 概率守恒和树结构检查。

### 3.8 恢复任务幂等演练

概率再分配通过持久化账本提交。每个原始路径 ID 在账本中最多提交一次；重复投递同一消息会返回 `committed=false`，不会再次增加代表质量。

演练会：

1. 提交约三分之一项目后模拟崩溃；
2. 原子 checkpoint 已落盘；
3. 从 JSON 账本恢复；
4. 重新投递完整批次；
5. 校验总质量、各代表质量和概率守恒；
6. 再完整投递一次，证明新增提交数为 0。

Checkpoint 使用临时文件、`flush/fsync` 和 `os.replace` 原子替换。

---

## 4. 数据格式

### 4.1 JSON 结构

最小结构：

```json
{
  "name": "示例情景集",
  "periods": ["T1", "T2", "T3"],
  "dimensions": [
    {"key": "demand", "label": "需求", "unit": "千件", "scale": 10},
    {"key": "freight", "label": "运价", "unit": "千元/千件", "scale": 3},
    {"key": "disruption", "label": "中断", "unit": "级", "scale": 1}
  ],
  "paths": [
    {
      "id": "A1",
      "probability": 0.5,
      "tags": ["normal"],
      "values": [
        [10, 6.0, 0],
        [20, 6.5, 0],
        [28, 7.0, 0.1]
      ]
    },
    {
      "id": "B1",
      "probability": 0.5,
      "values": [
        [10, 8.0, 0.8],
        [42, 8.0, 0.8],
        [48, 8.5, 1.0]
      ]
    }
  ]
}
```

`values` 的规范形状是：

```text
path × period × dimension
```

维度的 `scale` 用于把需求、运价、中断等级等不同业务量纲转成可相加的标准化距离。若缺少 `scale`，会使用稳健/IQR、标准差或极差回退并在尺度报告中说明。

`planning` 字段可选；缺失时使用 `scenario_tool/sample.py` 中的默认库存模型参数。

### 4.2 宽表 CSV

支持：

```csv
id,probability,t0_demand,t0_freight,t0_disruption,t1_demand,t1_freight,t1_disruption
A,.5,10,6,0,20,6.5,0
B,.5,12,6.4,.2,22,7.0,.5
```

期间索引从 `t0` 开始；维度名从列名中自动解析。完整示例见：

```text
examples/wide_paths_example.csv
```

### 4.3 长表 CSV

也支持：

```csv
id,probability,period,dimension,value
A,.5,T1,demand,10
A,.5,T1,freight,6
A,.5,T2,demand,20
```

### 4.4 示例文件

```text
examples/small_prefix_tree.json          # 8 路径前缀/尾部/完全搜索实例
examples/fictional_paths_120.json        # 120 条虚构多期路径
examples/wide_paths_example.csv           # 宽表 CSV 导入样例
```

---

## 5. 核心模型

### 5.1 业务与时间距离

对路径 `i` 和代表 `r`，先按维度业务尺度标准化：

```text
Δ[i,r,t,d] = (x[i,t,d] - x[r,t,d]) / scale[d]
```

再结合维度权重和期间权重计算分期距离。

前缀累积模式下，早期阶段差异会在之后每个前缀中重复出现：

```text
effective_stage_weight[s]
  ∝ importance[s] × Σ_{u≥s} importance[u]
```

这与信息揭示结构一致：第 1 期就分叉的路径，比最后一期才轻微不同的路径距离更远。

### 5.2 联合目标

给定代表集合 `R`，每条原始路径的叶子成本由以下部分组成：

```text
C(i,r) =
    distance_weight × probability_weighted_business_distance(i,r)
  + tailWeight × asymmetric_loss_error(i,r)
  + tailConditionalWeight × conditional_tail_group_penalty(i,r)
  + tiny_deterministic_tie_break
```

其中非对称损失代价对“代表损失低于原路径损失”的尾部低估施加更高惩罚；`tailGuards` 则把最高损失的若干正概率路径强制纳入代表集合。

代表概率不是简单平均，而是按实际分配到该代表的原始概率质量求和：

```text
q[r] = Σ_{i: assignment(i)=r} p[i]
```

因此总质量守恒：

```text
Σ_r q[r] = Σ_i p[i]
```

### 5.3 前缀 DP

对原始前缀树节点 `u` 和缩减前缀树节点 `v`，在相同深度计算：

```text
DP_d(u,v) =
  Σ_{u_child} min_{v_child} DP_{d+1}(u_child, v_child)
```

叶子层接入联合叶子成本。回溯 DP 即得到给定代表集合下的全局最优前缀兼容映射。独立检查器不依赖 DP 自身，会重新扫描共同前缀组并报告违规。

### 5.4 优化方法

- 小实例：`itertools.combinations` 完全枚举；
- 中等实例：确定性 PAM/k-medoids++ 风格初始化、medoid 更新和 O(N) swap-delta 优化；
- 超过 500 候选：CLARA 风格分层抽样，在至多约数百条路径上递归优化，映射回 medoid 后再做一次全数据 swap 精炼；
- 高损失守卫和正概率候选限制贯穿所有方法。

大规模启发式不声称全局最优；全局最优性由小实例完全搜索对照验证，而概率守恒、前缀兼容性和树结构在所有规模上都是硬检查。

---

## 6. HTTP API

服务由标准库 `http.server.ThreadingHTTPServer` 提供，静态目录为 `web/`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/sample?size=180` | 确定性虚构示例 |
| GET | `/api/sample?toy=1` | 8 路径小实例 |
| POST | `/api/import` | 导入 JSON/CSV 文本 |
| POST | `/api/validate` | 规范化、合并重复并检查概率 |
| POST | `/api/analyze` | 运行缩减、树、风险和规划比较 |
| POST | `/api/pareto` | 运行多权重代价前沿 |
| POST | `/api/exact-check` | 完全搜索和启发式对照 |
| POST | `/api/recovery-drill` | 崩溃恢复幂等演练 |

错误响应：

- Schema/概率验证错误：HTTP `422`，带结构化 `report`；
- 其他请求格式或参数错误：HTTP `400`；
- JSON 输出禁用 `NaN`/`Infinity`。

---

## 7. 代码结构

```text
scenario_tool/
  dataset.py      # schema 规范化、重复合并、零/负/未归一化检查
  io.py           # JSON/宽 CSV/长 CSV 导入
  distances.py    # 业务尺度、维度权重、分期/前缀累积距离
  risk.py         # 加权分位数、VaR、CVaR、损失尺度
  tree.py         # 前缀索引、前缀 DP、情景树构建与校验
  reduction.py    # 联合目标、概率再分配、完全搜索、启发式/大规模优化
  planning.py     # 缩减前后共用的多期库存/应急采购模型
  pipeline.py     # analyze / pareto / exact / recovery 应用编排
  jobs.py         # 持久化概率账本、崩溃恢复、幂等提交
  sample.py       # 确定性虚构数据和 8 路径对照实例
  server.py       # 零依赖 HTTP API 和静态文件服务
  cli.py          # validate/reduce/exact-check/recovery-drill/serve
web/
  index.html      # 中文工作台页面
  styles.css      # 响应式 light/dark 样式
  app.js          # 原生 JS 状态、API、SVG 图表、树和表格
tests/
  test_scenario_tool.py
examples/
  ...
```

---

## 8. 已验证的性质

当前测试覆盖：

- 重复路径概率先合并；
- 零概率与未归一化分别报告；
- 显式归一化后的概率守恒；
- 负概率拒绝；
- 8 路径实例枚举全部 56 个子集；
- 启发式匹配该小实例完全搜索最优；
- 故意构造的提前分叉分配会被前缀检查器识别；
- 缩减树概率和、父子链接和叶节点数有效；
- 零概率路径仍被分配但不能当代表；
- 规划样本外遗憾非负；
- Pareto 同时报告距离代价和尾部低估；
- 恢复演练守恒、匹配干净分配且重复投递零新增；
- checkpoint 中断后可恢复；
- 宽表 CSV 导入；
- 10,000 路径端到端运行满足代表数、`<1e-8` 守恒误差、前缀兼容和有效树。

最终回归命令：

```bash
node --check web/app.js
python3 -m unittest discover -s tests -v
```

---

## 9. 设计边界

- 所有业务数据均为**虚构数据**，不对应真实企业、人员或事件。
- 规划模型刻意保持小而可枚举，重点是用同一个决策模型比较原始/缩减集合，而不是声称它是完整供应链系统。
- 对上万路径使用确定性启发式和抽样精炼；小实例完全搜索负责提供最优性对照。
- 网页图表同时提供表格，颜色不是唯一信息通道。
- 项目没有 npm 包、打包步骤或第三方 Python 服务依赖。
