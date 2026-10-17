# RCCI02 冻结预测器：六根有界离线审计

2026-10-17 已完成 `rcci02_bnd_v1`，不要重复该 run ID。
实际结果及停止点见 [验收记录](../../reports/rcci02_frozen_audit_acceptance_20261017.md)。
下方保留完整流程说明，不表示应立即再次运行或进入训练。

工作目录 `E:/Prediction_RL_RCCI`，只读基线 `E:/Prediction_RL`。
配置 `configs/development/rcci02_frozen_audit_v1.json`，独立可读，无 extends。
本次入口只允许固定 train/validation 前三根，各根保留五候选；不支持 full、
改阈值、换 seed 或按结果过滤。不是 SUMO、正式评价或训练请求。

## 数据和加载

先验证已完成 RCCI01 full 的报告/receipt，再验证冻结数据 review、所用 root
的 collection receipts 和原始 c0–c4、训练 request/report/member/epoch receipts。
严格复用 ordinary/conditional 原三成员选定 ensemble，CPU、float32、单线程，
总共 12 次 ensemble 前向（每个前向内三成员），不重新选择 epoch。
输入模型的只有重新由观测历史建立的 tensors 和固定探针；未来仅用于离线比较。
校准/test target 文件不打开；不使用校准量修改接口或输出概率。

原 Raw/rollout/model/source 不修改，RCCI02-A 接口契约保持版本与半径不变。
只有新离线审计库/CLI、配置、测试和文档。运行前工作树必须干净。

## 命令

run ID 必须不存在；已运行 ID 不重试、不覆盖：

```powershell
python -B tools/audit_rcci_frozen_predictors.py prepare --baseline-root E:/Prediction_RL --run-id rcci02_bnd_v1
```

复制打印的实际 hash：

```powershell
python -B tools/audit_rcci_frozen_predictors.py run --request artifacts/rcci02/rcci02_bnd_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

若需一次执行 prepare+run（仍限同样六根）：

```powershell
python -B tools/audit_rcci_frozen_predictors.py audit --baseline-root E:/Prediction_RL --run-id FRESH_SHORT_ID
```

本机 Python 为 pytorch 环境；不需要复制资产、编译 st_cy、启动 TraCI 或修改依赖。
权重使用 `weights_only=True` 严格加载，拒绝缺 key、shape/dtype 不符或不符选定成员。

## 产物与指标

独立输出 `artifacts/rcci02/<run_id>/`：request、preparation、started、roots、report、receipt。
绑定实际 Git/source/config/environment、模型 SHA/选定 epochs、collection seeds、
完整 adapter contract 和所用文件 hashes；本轮没有新的 optimizer seed。
prepare 是准备状态，最终看 report.status。失败也写出原因并保留目录；负面科学
结果正常 complete，不能通过调整半径/删除 actor 伪造通过。

指标定义冻结如下：

- 25 个未来点的个体投影覆盖与原因、完整 actor/ego-actor 支持；不把 root 算为
  预测准确命中。原生/模型完整性与截尾分开。
- 149 维摘要中完整 following/cross/overlap/margin 候选数量，保留分母 5×roots。
- 原生已观测连续前缀（终止点排除、车辆消失即截断），两模型共同有效点上的
  progress 误差；两模型共同完整可投影前缀上的窗口评价。
- 进入/离开 observed-by-T 的 TP/FP/FN/TN，matched event 时间误差，实际观察
  时域内的 overlap 误差；只有完整非左截尾窗口才评价 signed margin/进入次序。
  MAE 缺少支持时为 null，不用 0 代替；只在都预测到事件时计算的时间误差不能
  代替漏报/误报。提前结束的 T 不当成完整 5 秒。
- 自车原参考 rollout 的投影支持/observed progress 误差单列，避免把共同自车误差
  归因为邻车预测器。
- 已完整观察到的未来控制体积占用窗口，是否完全位于 Raw 四采样时刻之间。
  这是几何事件描述，不是物理碰撞漏检率。
- 相对零 jerk 候选的邻车 XY/speed 变化，以及固定邻车为零 jerk 响应、保持所有
  自车探针时的摘要变化。替换可 OOD；只表征接口敏感性，不证明策略利用、因果
  正确性或控制收益；分开报告连续值变化与 mask 变化。

所有几何真值是**已知 lane progress + 同一 road-aligned 控制体积近似**，不是
原生 heading footprint 碰撞真值。覆盖不全的样本保留在分母/缺失统计，不用
只报容易投影部分的误差掩盖问题。统计单元是 episode；actor/candidate/time
不是独立场景，六根前缀不能代表整个 train/validation。

## 停止条件与下一步

本次不设必须让 B3 胜过 B2 的效果 gate。若覆盖低、误差支持很小或候选差主要
变成 mask 差异，报告这些事实并先审阅；不自动放宽接口、升级模型或启动 DDPG。
只有审阅离线结果后再决定是否需要扩展冻结离线审计或进入 RCCI03 有界原生接入。
本命令不允许进入 RCCI04 正式训练，也不覆盖历史 RCCI01 请求。
