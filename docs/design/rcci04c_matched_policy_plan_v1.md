# RCCI04c：匹配策略实验方案（待人工批准）

2026-10-18。RCCI04b已通过；这是下一阶段实现的明确方案，不是训练授权，不表示
正式CLI或请求已存在。当前不再追加诊断场景，不启动训练或长评价。

## 1. 只研究一个2×2问题

| 组 | 预测器 | 接口 | 策略来源 |
| --- | --- | --- | --- |
| Raw-B2 | 冻结ordinary三成员 | 原149维摘要 | 复用原120k实验权重与各节点 |
| Raw-B3 | 冻结conditional三成员 | 原149维摘要 | 复用原120k实验权重与各节点 |
| RCCI-B2 | 同一ordinary三成员 | 冻结v2、严格等价cached map | 新策略从头训练 |
| RCCI-B3 | 同一conditional三成员 | 冻结v2、严格等价cached map | 新策略从头训练 |

预测器、actor/history/mask/omitted设计、五探针、三成员均值先于后处理均不变。
不训练预测器，不加多模态/成员级后果聚合、地图网络、奖励、commitment或Shield。
不增加No-Forecast/zero等训练组、不加入作者50秒协议、泛化或新测试场景。

每组原观测20维+特征149维；ALL再追加一个TimeFeature。保持相同MLP参数量。
RCCI包含显式地图/车身/全时域后果与validity先验，不能把差异只称为纯编码效果。
短支持前缀、无效后缀、root-only状态原样保留；不删除难actor或改v2来追求收益。

## 2. 训练条件与预算提案

- 两组新RCCI × optimizer seeds 0/1/2：**6次训练，720,000请求步**。
- 每实例请求120k，按原完整episode边界保存，最多120499实际步；不从旧policy续训。
- 保留20k/40k/60k/120k四节点；120k是唯一主终点，中间节点仅诊断，不选最佳节点。
- CPU、torch_threads=1、训练worker=1。原连续DDPG、原奖励/结束/bootstrap/replay不变。
- 与原preset一致：lr_pi=lr_q=0.0002，gamma=.98，batch=100，replay_start=5000，
  buffer=1,000,000，polyak=.005，noise=.1，update_frequency=1，原2M scheduler不变。
- 原持续交通reset：每训练job新进程，episode之间不重启SUMO/不重置交通RNG。
  不切换fresh-task课程；控制周期.2 s、20 s预热、连续jerk[-5,5]保持。
- `simulation_blocking_exact_v1`：预测完成后推进仿真，耗时只记录，不因0.5s丢弃结果。
- 同seed新RCCI初始policy/q参数应与历史expanded Raw初始化一致，未通过则停止核查。

六个旧Raw最终权重与四节点SHA已有RCCI04a证据，运行身份有RCCI04b限定桥接。
实际运行配置须使用与旧contract字节SHA相同的已核验原配置资产，而不是改写旧SHA。
只复用权重，不把weights-only checkpoint当完整训练恢复点。

## 3. 评价与统计提案

沿用冻结开发场景200–219，Gym20、新进程首回合、无exploration。全部3个训练seed保留。
两组新RCCI × 3 seeds × 4节点 × 20场景 = **480个新RCCI评价回合**。

旧Raw已有480个封存回合作为历史回归参照。新匹配请求下由用户执行当前环境的
同480个Raw评价（不重新训练），精确核对旧轨迹/动作/结果，避免将未实测的其余
场景/节点标成已在新环境验证。共960个当前环境评价调用，但只有**20个交通场景**。
这是同一2×2的既定节点/场景，不是额外泛化实验；长评价由用户执行。

主比较：120k的RCCI-B3相对Raw-B3。次要：RCCI-B2相对Raw-B2、各接口内B3相对B2。
交互差值为 `(RCCI-B3−Raw-B3)−(RCCI-B2−Raw-B2)`，作为小样本描述性机制证据。
先同optimizer seed/simulator seed配对，再等权汇总训练seed，报告全部单seed与离散度。
不把多节点、多模型、多个候选或多步当独立样本；3训练seed不支持强收敛/普适性结论。

保留原reward return、自然到达/ego碰撞/期限失败、完成时间、实测jerk、低速与饱和、
以及RCCI支持长度/失效原因/actor覆盖。成功条件效率与全回合结局分开，不能只报成功组。
当前任务若饱和或RCCI负效果，正常聚合，明确保留/放弃结论，不自动延长到200k。

## 4. 实现验收边界

新增独立RCCI训练适配/contract，明确区分Raw与RCCI模式。旧preset/loader继续拒绝
把Raw权重装成RCCI；不修改共享旧训练函数或历史请求guard。
优先复用原DDPG/训练循环，仅显式替换接口及审计依赖，不另写DDPG算法。

先完成单元回归、严格checkpoint/save-load、无未来泄漏、初始化/forward/终止检查，
再另行冻结有上限的入口smoke；不把入口smoke称作正式训练。正式prepare必须干净提交，
保存完整resolved config/commit/seed/命令/环境/模型/父证据哈希与不可变receipt。

工程错误与方法负结果分离：负效果不能触发异常阻止聚合；加载/数值/协议错误必须
显式停止并保留失败，不删seed、不覆写报告、不静默部分加载或使用stale。

批准该方案后才实现并交接实际正式命令；训练、长评价仍由用户手动执行。
