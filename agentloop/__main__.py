import sys

from .agent import run


def main():
    if len(sys.argv) < 2:
        print('usage: agentloop "<task>"')
        sys.exit(1)
    answer, log = run(" ".join(sys.argv[1:]))
    print(answer)
    print(f"[log: {log}]")


if __name__ == "__main__":
    main()
