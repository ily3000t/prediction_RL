# RCCI04b：Raw 历史对照的有界运行时桥接

2026-10-18，桥接实现/运行前冻结。承接 RCCI04a，不改原预测器、Raw 特征、
DDPG、奖励/动作/终止、RCCI v2 或任何历史请求守卫，不训练、不做方法排名。

## 差异与选择

RCCI04a 已完整核验六个120k Raw训练实例及480个既有评价。当前 extension 与历史
SHA不同，但83个执行源码、25个上游文本、安装的ALL源码及非extension环境一致。
配置仅Git换行不同。桥接不应通过重写旧contract、复制binary或重训来掩盖差异。

- 读同一只读baseline的原配置文件，其实际字节SHA必须等于旧权重contract记录；
  不拿当前CRLF字节冒充旧SHA。两种binary均使用这个原配置及当前未改的上游源码。
- 两种binary由独立worker按绝对路径显式加载：历史baseline pyd、当前本地构建pyd。
  必须核对实际加载路径和SHA；禁止静默fallback、构建、安装或复制。
- 使用原 `train_matched_ddpg.evaluation_job` 的同一code object；只注入审计session、
  明确的只读资产解析和透明记录器，不执行旧run/load_request或放宽其守卫。
- eval只加载严格完整的policy/q，replay和优化器状态必须保持空、权重/预测器不变。

## 冻结的有界预算

六个最终Raw：ordinary/conditional × optimizer seeds 0/1/2，固定120k主终点。
只用既有开发列表的首个simulator seed=200，不依据旧结果选择场景/模型。
每模型分别执行historical/current binary，各新进程一个Gym20首回合，串行运行。

| 项目 | 上限 |
| --- | ---: |
| sessions/episodes | 12 / 12 |
| 每回合控制步 | 500（原始终止，不能为通过而截短） |
| 总控制步 | 6,000 |
| 总simulationStep | 7,224 = 12 × (100预热+1插入+500控制+最多1期限清理) |
| ensemble/member forward | 6,000 / 18,000 |
| policy forward | 6,012（每回合含一次terminal eval） |
| worker watchdog | 180 s |

CPU、torch_threads=1、固定五探针、simulation_blocking_exact_v1、无exploration，
其余完全沿用原120k评价契约。驾驶失败是合法结果，不决定工程gate。
超过预算、加载/数值异常、观测/轨迹不一致则停止并保留失败，不换seed或放宽容差。

## 必须核验

1. 每worker与对应历史seed200评价：完整actions/rewards/done、states观测哈希与交通
   记录、raw terminal sidecar、episode指标、resolved原配置与checkpoint全部精确一致。
2. 两binary之间额外逐步比较：169维完整观测与执行审计、RNG、实际policy输入170维
   （含TimeFeature及mask）；透明hook不得改变返回值或消耗RNG。
3. worker内对上游ST和extension公开入口增加仅诊断的调用guard：若被调用即记录并
   报错，不替换控制、不执行规划器。必须全部0调用。它不改源码或安装包，退出恢复。
4. terminal不再做预测；forward数量与真实非terminal观测一致。原env只由step推进。
5. 干净固定源码、原始已封存证据/配置/权重/binary哈希、预算和输出receipt完整。
   baseline源码/Git/产物保持只读；新目录一次写入，失败不自动重试。

## 结论边界与后续

通过仅证明这些冻结Raw策略在当前纯Gym/DDPG路径上的跨构建桥接及历史回归，
不证明两个Cython模块的全部MPC函数等价，不证明新接口有效或额外场景泛化。
不改变“binary字节身份不同”这个事实。

若通过，可在显式记录本桥接和原配置字节来源的RCCI04匹配协议中保留六个Raw权重。
其余开发场景的当前环境评价由用户在新协议下运行；不把12个工程回合当新方法效果。
新RCCI策略仍须从头训练，预算/正式运行需另行冻结并人工批准。
