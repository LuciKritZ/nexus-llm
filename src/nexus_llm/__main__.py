import argparse
import logging

import uvicorn

from nexus_llm.config import settings
from nexus_llm.services.cache import ImageCache


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t  %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="nexus-llm proxy")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the proxy on")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the vision API cache")
    args = parser.parse_args()

    if args.clear_cache:
        cache = ImageCache()
        cache.clear()
        print("Cache cleared!")
        return

    if not settings.proxy_password:
        import sys

        print(
            "CRITICAL ERROR: PROXY_PASSWORD is not set in your .env file",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run("nexus_llm.app:create_app", host="0.0.0.0", port=args.port, factory=True)


if __name__ == "__main__":
    main()
