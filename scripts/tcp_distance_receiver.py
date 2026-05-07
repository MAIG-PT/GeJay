"""Simple TCP receiver for streamed hand-distance averages."""

import argparse
import json
import socket
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.distances_config import TCP_CONFIG


def main(host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Listening on {host}:{port}")
        print("Waiting for tracker connection...")

        client, address = server.accept()
        with client:
            print(f"Connected by {address[0]}:{address[1]}")
            buffer = ""

            while True:
                data = client.recv(TCP_CONFIG["buffer_size"])
                if not data:
                    print("Connection closed")
                    break

                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON: {line}")
                        continue

                    print(
                        "frame={frame} interhand_average={interhand_average} thumb_index_average={thumb_index_average}".format(
                            frame=payload.get("frame"),
                            interhand_average=payload.get("interhand_average"),
                            thumb_index_average=payload.get("thumb_index_average"),
                        )
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Receive averaged hand-distance data over TCP"
    )
    parser.add_argument("--host", default=TCP_CONFIG["host"])
    parser.add_argument("--port", type=int, default=TCP_CONFIG["port"])
    args = parser.parse_args()
    main(args.host, args.port)