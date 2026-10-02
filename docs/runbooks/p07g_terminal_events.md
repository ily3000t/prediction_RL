# P7g 原始终止事件审计运行说明

项目根目录E:/Prediction_RL，激活pytorch；无训练步骤。
配置：configs/development/p07g_terminal_events_v1.json（独立可读、未知字段拒绝）。
入口：tools/audit_terminal_events.py。

## 开发者有界接入

```powershell
python -B tools/audit_terminal_events.py audit --run-id p7g_audit_v1
```

场景200、作者一份及B0种子2/B3种子0、四条件，12新回合；只读审计是否改变
原奖励/动作/轨迹，及原始碰撞/到达事件如何重叠。这是机制诊断，不是最好模型选择。
所有原始simulationStep含预热/插入/原清理上限8136次（12回合），控制调用上限6006次。
不重复同run ID，不覆盖旧产物；失败目录保留，不能直接自动重试。

## 准备完整请求（不启动SUMO）

```powershell
python -B tools/audit_terminal_events.py prepare --run-id p7g_diag_v1
```

要求已提交干净工作树及完成audit。原84格全部重评：作者1份、B0/B3各3份，
三个原场景200/210/219，Gym/作者各20/50秒预热。旧raw事件不存在，不能缓存复用。
prepare冻结完整解析配置/源码/依赖/权重/原轨迹及audit证据，不开启sealed test。

## 用户运行84回合

```powershell
python -B tools/audit_terminal_events.py run --request artifacts/p7g/p7g_diag_v1/request.json --confirm-request-hash <prepare打印的哈希>
```

两进程、CPU一线程、每scene独立新进程首回合。不要同时编辑代码/配置或运行训练。
若中断，只有完整且所有哈希/收据一致的回合可用同命令追加--resume复用；
未完成/失败目录显式停止，不删除或覆盖它们。run完成自动聚合。

## 产物与解释

- 每回合episode.json/trace.json：原生结果/轨迹，必须与旧P7f可观测部分严格一致。
- raw_events.json：每步前后ego及live IDs、arrived IDs、colliding IDs、碰撞对象、
  实际原vehicle.add到达位置参数、显式remove边界；在额外step清掉事件之前记录。
- analysis.json：独立ego事件分类与原标签差异。不是重新训练或重新计算奖励。
- aggregate.json：全部84格双口径事件计数，method_effect_gate=null。
- arrivalPos来源是实际捕获原add调用参数，非不存在的TraCI getter；路线是实际查询。
- natural_arrival要求原始arrived flag、没有记录到ego碰撞且终点一致；不是现实安全保证。
- 期限是仿真任务期限；背景碰撞、无法解释的移除不冒充ego碰撞/成功。
- 不合并跨driver奖励，不按结果挑协议，不根据单次负结果换seed、模型或阈值。

当前本说明是实现流程；测试/audit真实结果及精确完整请求哈希需验收后补充。
