# 银行远程坐席知识助手 · 多 Agent 版（v2）

面向银行远程银行中心坐席的多 Agent 知识助手：坐席接电话时需要在几秒内
答出准确的业务规则，答错是事故、编造是违规。v2 将原单 Agent 架构升级为
**LangGraph Supervisor 多 Agent 架构**——合规守卫、知识检索、业务规则、
工单处置四个专家 Agent 由 Supervisor 编排，答案可溯源、行为可评测。

> v1（单 Agent ReAct + 置信度路由）见本仓库 `legacy-v1` 分支，`main` 分支为 v2 多 Agent 架构升级版。

## 快速开始

```bash
# 环境：Python 3.10+，依赖见 requirements.txt
pip install -r requirements.txt

# 1. 重建知识库索引（data/chroma 已被 gitignore）
python scripts/build_kb.py

# 2. 命令行对话
python cli.py

# 3. 启动 API 服务
python -m src.server

# 4. 跑测试与评测
pytest tests/ -q
python evals/run_eval.py
```

## 文档

- [docs/01-产品设计书.md](docs/01-产品设计书.md) —— 业务场景、Agent 角色划分、验收标准
- [docs/02-技术设计.md](docs/02-技术设计.md) —— 编排图、状态契约、工具边界、评测口径
- [docs/03-验证报告.md](docs/03-验证报告.md) —— 端到端验证与评测结果

## 架构一览

```
坐席问题
   │
   ▼
[Guard 合规守卫]──拒答/强制转接──►[Ticket 工单处置]
   │ 放行                                ▲
   ▼                                    │（投诉/资金安全）
[Supervisor 编排]───显式规则路由─────────┘
   ├─► [Knowledge 知识检索 Agent]  RAG 证据包 + 置信度
   ├─► [Rules 业务规则 Agent]      费率表/规则表逐条判定
   └─► [Synthesis 答案合成]        带引用的可溯源答案
```

版本推进采用 git 分阶段提交（环境搭建 → 产品设计书 → MVP → 技术设计 → 验证收口），每阶段一个 commit，可回滚。
