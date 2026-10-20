"""LangGraph 图组装：Guard → Supervisor → {Knowledge, Rules, Ticket, Synthesis}。

图结构：
  START → guard → (REFUSE→synthesis | ESCALATE→ticket | PASS→supervisor)
  supervisor → (knowledge | rules | ticket | synthesis)
  knowledge → supervisor   (回环，带新证据再决策)
  rules → supervisor       (回环)
  ticket → END
  synthesis → END

回环由 supervisor 的 iterations 字段 + MAX_ITERATIONS 保护，防死循环。
"""
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.graph.guard import guard_node
from src.graph.supervisor import (supervisor_node, route_after_guard,
                                  route_after_supervisor)
from src.graph.knowledge import knowledge_node
from src.graph.rules import rules_node
from src.graph.ticket import ticket_node
from src.graph.synthesis import synthesis_node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guard", guard_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("rules", rules_node)
    g.add_node("ticket", ticket_node)
    g.add_node("synthesis", synthesis_node)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"supervisor": "supervisor", "ticket": "ticket", "synthesis": "synthesis"})
    g.add_conditional_edges("supervisor", route_after_supervisor,
                            {"knowledge": "knowledge", "rules": "rules",
                             "ticket": "ticket", "synthesis": "synthesis"})
    g.add_edge("knowledge", "supervisor")
    g.add_edge("rules", "supervisor")
    g.add_edge("ticket", END)
    g.add_edge("synthesis", END)

    return g.compile()


# 模块级单例（编译图开销不大，但避免重复）
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
