# RCCI03 原生接入：工程验收与限制

## 结论

**工程接入通过，方法效果未证明，也不据此启动正式 DDPG。**
固定脚本动作下，Raw/RCCI 与 baseline 的原 observation、动作、reward、done、
交通、control.delay、Python/NumPy/torch RNG 均完全相同；在线特征与同一缓存的
离线计算逐值相等，模型不更新。合法短前缀真实传给观察，而非变成异常或 stale。

同时发现两个实际成本：RCCI 后处理明显更慢；这三段脚本轨迹中的支持前缀比
先前离线 cohort 短。不能只报告 gate=true 而隐藏这两项，更不能称作方法胜过 Raw。

## 1. 先冻结，再实现和运行

- 规范提交：`e2a6b38`，`docs/design/rcci03_native_smoke_v1.md`。
- 实现/测试/配置提交：`c4b95fa14d280da14aadd195b4b66d06caeefc1c`。
- 请求：`artifacts/rcci03/rcci03_native_v1/request.json`。
- request hash：`17e149d31001ab4ba6aed4123af1bb209b114ae55a6b54b5190f3230b5c7acf6`。
- report SHA256：`f8ac303062a2ccd2a5e9e500dea80529c28e3ce5d8958d8e91daa2af074543f6`。
- index SHA256：`19aa5a965569e7870b3658250a9fcf28b00abc5611f7f453e3647503c7382290`。
- status=complete，engineering_gate=true；约 270.45 s。

配置：`configs/development/rcci03_native_smoke_v1.json`，未在观察结果后修改。
没有 agent、optimizer、policy checkpoint、学习或新采集数据集；两个冻结 ensemble
从只读 baseline 严格加载，没有部分加载、重选模型、reward/动作/seed 修改。

## 2. 原生资产

在当前工作树用原 `python -B setup.py build_ext --inplace` 构建，未复制 baseline
的二进制，不安装/升级依赖。setup/pyx 核对至原 75051acc 来源，仅允许 Git LF/CRLF。

| 项目 | 实际身份 |
| --- | --- |
| Python / NumPy / Cython | 3.10.16 / 2.2.6 / 3.0.12 |
| PyTorch / Gym / ALL | 2.5.1 / 0.26.1 / 0.5.3 |
| SUMO / traci | SUMO 1.22.0 / traci 1.25.0，沿用现有环境 |
| MSVC inventory | BuildTools2022 / 14.44.35207 |
| 本地 extension | `RL-MPC-LaneMerging-master/st_cy.cp310-win_amd64.pyd` |
| extension SHA256 | `aeacc8434bb6c0cd9777105fe21f8e97d5d52908379ef8e467d3d4c986acc04a` |
| native identity acceptance | `artifacts/rcci03_build/native_audit_v1/report.json` |
| acceptance SHA256 | `afd5bd0fec517d368180a99ac254847d4ff4664ea4535232db8ac00410ee771a` |

构建 exit=0 且新进程可从当前工作树精确导入。**原构建 stdout 捕获发生 GBK
UnicodeDecodeError，保存值为 null**；没有覆盖旧记录或重编译来隐藏缺口。
另作只读 binary/source/import 验收，记录 compiler inventory 原始 bytes；该 inventory
不是恢复的原编译命令。不能说完整编译日志已保存。后续 smoke worker 日志改用
二进制捕获，70 个产物文件全部 SHA 校验。

## 3. 实际预算和语义检查

| 项目 | 冻结上限 | 实际 |
| --- | ---: | ---: |
| 独立新进程 / SUMO session | 10 | 10 |
| episodes | 15 | 15 |
| 受控 step | 1,800 | 600 |
| 含原 reset warmup 的 simulationStep | 3,315 | 2,115 |
| ensemble forward | 1,444 | 484 |
| member forward | 4,332 | 1,452 |

每个 arm 的 stream 为两回合共 80 个受控步，mixed 为一回合 40 步。
stream 的两次原始终止分别为 upstream_arrival、upstream_collision；mixed 到 cap
时仍未 done，正常关闭，不伪造终止。固定动作不是学习策略，这些结局不得作方法比较。

8 项 learned-vs-baseline 原状态/动作/reward/终止/traffic/RNG 完全一致检查全为 true。
所有非终止预测帧均验证 simulation time 未推进；每步只由 base.step 执行原动作。
terminal 不读帧、不推理、扩展 block 全零；reset 历史清空后从 1 帧开始，上限 11。

逐次用原 forward 缓存离线重算，而非二次推理；36 个固定索引 0/1/10 的保存示例
另作独立重算，均逐值相等。Raw/RCCI 同模型的 18 对保存示例，其 history、mean、
ego rollout 也逐值相同，未发现新 provider 改变模型输入/输出的证据。
模型 state_dict、requires_grad、grad 均保持；本轮复核新增 forward=0。

## 4. 接入变化与明确未接入项

新增 `features/rcci_provider.py` 和 `envs/rcci_features.py`，复用冻结 v2 纯函数。
Raw provider、原 wrapper、预测器、DDPG/奖励/终止/动作实现完全未改。
RCCI mode 独立为 `rcci_ordinary/rcci_conditional`，metadata 明确版本和工程用途。
旧 Raw preset/checkpoint 会拒绝它们；149 维相同不是权重语义兼容。
工程错误使 wrapper faulted，不能继续 step/reset 或取 stale；合法模型倒退/离路
继续按未修改的 v2 前缀/mask 契约描述，不修平、不删除 actor。

尚无 RCCI 正式训练入口、权重契约、学习曲线或效果报告。不能将 smoke 工具用作
训练 CLI，也不能把旧 Raw 策略直接接新 observation 当兼容性验证。

## 5. 真实支持：必须保留的负面发现

两种 RCCI 各有 121 次非终止观察、605 个 probe 候选，来自相同三段脚本。
候选不是 605 个独立场景；也不是正式策略或全训练分布的覆盖率估计。

| following 支持 | B2-RCCI | B3-RCCI |
| --- | ---: | ---: |
| 完整 5 s | 33/605（5.5%） | 40/605（6.6%） |
| 仅 root、无未来支持 | 15/605（2.5%） | 62/605（10.2%） |
| 共同前缀中位数 | 1.2 s | 0.4 s |
| 每次观察最短候选前缀中位数 | 1.2 s | 0.2 s |

这是先前离线 cohort 无法替代的在线分布检查，不能把离线完整覆盖直接外推过来。
失效计数：两组 ego endpoint 均 94；actor route reversal 为 4,850 / 4,114，
off_path 为 15 / 36。计数含多 actor/候选重复，不是独立危险事件。
Raw/RCCI 的保存预测相同，说明不能把这些短前缀简单归咎于新 wrapper 改了预测值。

按已有历史 mask 定义分组，不新选阈值：

| 组别 | 候选数 | B2 中位前缀 / root-only | B3 中位前缀 / root-only |
| --- | ---: | ---: | ---: |
| history<11，30 次观察 | 150 | 0.4 s / 15 | 0.2 s / 52 |
| history=11，91 次观察 | 455 | 1.2 s / 0 | 1.4 s / 10 |

历史不足组的支持更短，但车辆位置/自车状态也同时改变，不能单独归因为冷启动。
也不能说“补齐历史即可解决”：完整历史时仍多为部分前缀。
这不违反 v2 已冻结的合法缺失语义，也不事后修改工程 gate；但足以限制训练准备度
和方法主张。当前未将所有短前缀具体归因于训练覆盖、交互表达或道路池化。

## 6. 描述性计算成本

下表来自 wrapper 的计算计时，stream 含两个 terminal 零 block（82 次观察/80 forward），
mixed 为 41 次观察/forward；不含外部离线重算检查。只作 smoke 描述，不是独立部署评测。

| provider | stream 平均 / 最大（s） | mixed 平均 / 最大（s） |
| --- | ---: | ---: |
| Raw ordinary | 0.0191 / 0.0344 | 0.0227 / 0.1642 |
| Raw conditional | 0.0231 / 0.1917 | 0.0229 / 0.1528 |
| RCCI ordinary | 0.2081 / 0.3464 | 0.2943 / 0.4364 |
| RCCI conditional | 0.2111 / 0.3783 | 0.3131 / 0.4298 |

RCCI 明显较慢。阻塞契约让它不改变 observation，但会增加训练成本。
未设置 0.5 s 成败门槛，未以 wall-clock 丢弃结果；不能由本表宣布软/硬实时部署可行。
正式大预算前值得考虑静态地图几何缓存等**严格语义保持**优化，并用冻结轨迹做逐值
等价测试；没有授权以减少 actor/点数、缩短预测时域或换种子代替优化。

## 7. 测试、阶段决策与下一步

- 首轮接入/Raw/原生环境选定测试 54 passed。
- 最终选定离线+原生回归 **356 passed，44.43 s**，无 deselect。
- 此前缺少 st_cy 的四项用例现已执行通过；仍只称选定回归，不称全仓测试通过。
- 测试目录 `artifacts/tests/rcci03_reg_v1`；70 文件 SHA、8 项 parity、36 个缓存例
  和18 对 Raw/RCCI 相同预测的独立只读复核均通过。

RCCI03 工程工作已完成。**不建议立即启动六次/十二次正式大预算训练**：
本次支持长度和计算成本提示需要更审慎的准备，但也不应据三段脚本直接否定方法。
如果继续，优先做 RCCI04 的 Raw 复用审查和预算设计；可用现有缓存做一次有界的
计算剖析/严格等价优化，不重新收集大数据或不断跑 smoke 挑好结果。
若要修改前缀/池化语义，必须另立版本、在新运行前冻结，不能回写 v2 或本报告。
任何学习性验证应四组匹配、固定预算、全部 seeds 保留，并经用户批准后自行执行。

未合并、推送或打 tag；未修改只读 baseline、旧项目或原离线结果。native二进制、
生成 C、日志和大数据均被忽略，不进入 Git。本阶段不是可运行方法效果里程碑。
