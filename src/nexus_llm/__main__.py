import argparse

import uvicorn

from nexus_llm.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="nexus-llm proxy")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the proxy on")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the vision API cache")
    args = parser.parse_args()

    if args.clear_cache:
        print("Cache cleared!")
        return

    uvicorn.run("nexus_llm.app:create_app", host="0.0.0.0", port=args.port, factory=True)

if __name__ == "__main__":
    main()
