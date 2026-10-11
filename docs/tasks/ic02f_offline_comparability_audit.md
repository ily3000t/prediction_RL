# IC02f：共同时间窗、关键角色与 D 道路语义核查

## 目标与前置

用户在 IC02e 完成后明确批准继续纯离线核查。输入仅为封存的原三任务及剩余 29 项，共 32 train tasks、96 roots；不访问 predictor test 或策略开发评价结果。原 5 秒 I、旧 IC01 尺度、所有源报告和失败记录保持不变。

## 允许与禁止

允许独立审计实现、离线 fixture、已有收据完整性与科学指标重算、命名审计产物和本地原子提交。禁止 SUMO、模型加载/推理、数据补采、训练、课程赋层、阈值拟合、pi 冻结、推送、合并或 tag。旧主项目只读。

## 指标与验收

- 预声明网格为 5/10/15/25 ticks，即 1/2/3/5 秒。各窗口必须五候选、全部选定稳定 ID、每个 tick 都有非终止观测；缺失为 unavailable，不能补零或换任务。
- 独立报告全 96 roots 的最长共同 dense 前缀的最小值。这是依据观测支持得到的探索性上限，不是优化策略收益的选窗，也不是已冻结的新训练合同。原始可变有效支持的 5 秒 I 原样保留。
- 同窗口比较原关键集合、去重的目标前/后车集合及逐 actor 响应。角色不因未来改变；不把 ego 响应计入 I。展示加入额外 actor 的平均稀释与排序敏感性，不预设第三辆车一定无用。
- 保留原 D 字段，但分开 route-projected gap/TTC、同路径 gap/CV-TTC 和跨路径 merge ETA 代理。欧氏前保险杠距离不是车辆表面间距；投影负间距不作为即时碰撞证明。没有新安全界或 D 加权分数。
- 正常聚合零响应、截尾及不可计算窗口；工程完整性与课程科学有效性分开。
- 启动前及收尾绑定源请求/报告/收据/科学资产 SHA，全部 32 任务重算旧 evidence，原 RNG 和输入保持不变。实验来源 commit 和 resolved config 被记录。

## 产物与命令

配置：`configs/development/ic02f_task_comparability_audit_v1.json`；模块：`curriculum/task_audit.py`；工具：`tools/audit_curriculum_tasks.py`。输出仅 `artifacts/ic/ic02f_audit_v1/`，含 request、started、roots、report、postflight；完成或失败均不可覆盖或自动重试。

在审查、测试和本地实现 commit 后运行：

```powershell
python -B tools/audit_curriculum_tasks.py prepare --baseline-root E:\Prediction_RL
python -B tools/audit_curriculum_tasks.py run --request artifacts/ic/ic02f_audit_v1/request.json --confirm-request-hash <prepare打印的hash>
```

## 暂停与交接

源证据不一致、任务缺失、无共同有效前缀时保存失败并暂停，不降低标准。完成后提出 task-level D/I 规则与明确命名的有限池目标分布供人工审查；本阶段不实施分类或训练。固定步长的 32 任务不宣称是原 continuous-traffic 目标分布的无偏样本。

## 实际状态

已在实现 commit `24b908d…` 完成命名请求，32 项全部通过，0 新会话/推理/训练。470 项离线测试通过；共同 dense 前缀为 14 ticks，独立直接公式复核 480 个窗口一致。原指标与基线未改。详细发现、文件哈希与剩余设计边界见 `reports/ic02f_offline_comparability_acceptance.md`；仍未赋层或放行训练。
