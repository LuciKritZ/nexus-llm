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

    if settings.proxy_password:
        import getpass
        import sys

        entered = getpass.getpass("Enter proxy password: ")
        if entered != settings.proxy_password:
            print("Incorrect password. Exiting.")
            sys.exit(1)

    uvicorn.run("nexus_llm.app:create_app", host="0.0.0.0", port=args.port, factory=True)


if __name__ == "__main__":
    main()
