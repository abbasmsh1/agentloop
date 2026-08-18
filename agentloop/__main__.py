import sys

from .agent import run


def main():
    if len(sys.argv) < 2:
        print('usage: agentloop "<task>"')
        sys.exit(1)
    answer, store = run(" ".join(sys.argv[1:]))
    print(answer)
    print(f"[log: {store.path}]")


if __name__ == "__main__":
    main()
