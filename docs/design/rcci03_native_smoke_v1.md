# RCCI03：独立原生接入与有界语义 smoke

2026-10-18，完整离线复盘后、原生构建/执行前冻结。用户“继续”授权本阶段接入与
有界 smoke，不授权正式数据/训练、模型修改或方法效果评价。

## 冻结边界

- 原 SUMO 1.22.0、训练配置 `configs/train_default_1.json`、奖励、终止、连续 jerk、
  0.2 s 步长、20 s reset warmup、持续交通与两个预测器全部不变。
- 新建 RCCI 专用 provider/wrapper；Raw provider、默认 wrapper、DDPG/权重契约不改。
- RCCI 模式单独命名，拒绝旧 Raw trainer/preset/checkpoint 将其当旧语义直接使用。
  本阶段没有 DDPG agent 或优化器，不用任何策略权重产生动作。
- v2 全部 149 字段保持不变，短前缀/无效后缀是合法 mask，不平滑、删除车辆或
  用 stale 替代；NaN/模型错误/非法 root 显式停止。终止 observation 继续零预测块。
- 不把宽节点 overlap 当碰撞，也不以 smoke 通过声称 RCCI 胜过 Raw。

## 原生资产验收

仅在当前 RCCI worktree 用原 `setup.py build_ext --inplace` 构建 st_cy，记录原 pyx、
setup、生成文件、extension、Python/NumPy/Cython/compiler/SUMO 身份。已有 MSVC 与
pytorch 环境可用，不安装/升级包，不复制旧项目二进制，不修改只读基线。
构建失败则保留日志并暂停原生运行，不静默切换 Python planner 或 USE_CYTHON。

## 确定的原生预算

每个 replay/arm 使用新进程、独立 SUMO session，父进程串行调度并统一封存。
arms：baseline、raw_ordinary、raw_conditional、rcci_ordinary、rcci_conditional。

| replay | simulator seed | 固定 jerk 循环 | episodes | 每 episode 决策 cap | 必须自然终止 |
| --- | ---: | --- | ---: | ---: | --- |
| stream | 0 | [5] | 2 | 160 | 是（保留任一原始终止原因，不筛选成功） |
| mixed | 100 | [-5,-2.5,0,2.5,5] | 1 | 40 | 否（到 cap 正常关闭，不伪造 done） |

总计最多 10 个 session、15 个 episode、1,800 次受控 step；每次 reset 原生
100 warmup + 1 insertion step，合计 simulationStep 上限 3,315。
原生 time-limit=500，不会在上述 cap 内触发其额外清理 step。
每个 learned arm 最多 320+41=361 次预测，四组总上限 1,444 ensemble forward。
最多 4,332 member forward。终止不预测；离线重算同一缓存不新增 forward。
每 worker watchdog=600 s；超出即失败，不删 seed、不增加 cap 重试。

## 必须通过的工程检查

1. 同 replay 所有 arms：原 observation 前缀、请求/实际执行审计、reward、done、
   traffic、control.delay 与 Python/NumPy/torch RNG 完全一致，不比较 wall-clock。
2. 预测/扩展帧读取前后 simulator time 相同；仅 base.step 推进模拟。
3. 用同一 provider 缓存的均值/ego/history 离线重算特征，必须逐值相等，不二次推理。
4. 历史上限 11、reset 清空、terminal 不读取/不推理且预测块全零；RCCI 输出 float32
   149 维，完整 observation=20+149，Raw 同维但名字/版本不同。
5. predictor 参数/buffers、requires_grad、grad 和 CPU 状态不变；候选布局固定，
   没有 reward、标签、真实未来、controller/MPC 接管或策略动作进入预测特征。
6. 保存实际步数/终止、首帧与逐帧信息支持、latency 描述、模型与构建身份，封存
   输入/输出 SHA。负驾驶结局不是 gate 失败；语义不一致、越预算才是工程失败。

请求须在干净已提交源码上准备，严格绑定父离线报告、当前源码/环境/配置/地图与
模型选择、原生 binary/build 输入。不重跑/放宽旧请求 guard。已启动目录不覆盖，
失败保留，无自动重试。只为工程 smoke 编写入口，不扩展正式训练命令。

## 后续决策

通过仅表示具备接入条件。RCCI04 Raw 复用与匹配训练仍另行冻结、人工批准、用户运行。
不改已冻结 v2 字段来追求更好的 smoke 指标；不加入新模型、reward、Shield 或新 seeds。
