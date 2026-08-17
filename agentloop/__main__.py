import sys

from .agent import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m agentloop \"<task>\"")
        sys.exit(1)
    answer, log = run(" ".join(sys.argv[1:]))
    print(answer)
    print(f"[log: {log}]")
