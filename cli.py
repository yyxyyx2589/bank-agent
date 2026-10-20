"""命令行入口：python cli.py "你的问题" """
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import get_app


def main():
    if len(sys.argv) < 2:
        print("用法: python cli.py \"你的问题\"")
        print("例: python cli.py \"信用卡分12期手续费多少\"")
        sys.exit(1)
    query = sys.argv[1]
    app = get_app()
    result = app.invoke({"query": query, "messages": [], "trace": []})

    print("=" * 60)
    print(f"问：{query}")
    print(f"路由徽章：gate={result.get('gate','?')} conf={result.get('confidence_level','-')}"
          f" iters={result.get('iterations',0)}")
    print("-" * 60)
    print(result.get("answer", "（无答案）"))
    if result.get("citations"):
        print(f"\n引用：{', '.join(result['citations'])}")
    print("-" * 60)
    print("执行轨迹：")
    for step in result.get("trace", []):
        print(f"  [{step['agent']}] {step['action']}  {step.get('detail','')}")


if __name__ == "__main__":
    main()
