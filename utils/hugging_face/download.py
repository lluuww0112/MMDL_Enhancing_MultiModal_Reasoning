import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into ./weights/<target>."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Hugging Face repo id to download, for example: liuhaotian/llava-v1.5-7b",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Prefer setting HF_TOKEN in the environment to avoid exposing it in process lists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path("./weights") / args.target
    token = args.token or os.environ.get("HF_TOKEN")
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.target,
        local_dir=destination,
        token=token,
    )

    print(f"Downloaded '{args.target}' to '{destination.resolve()}'")


if __name__ == "__main__":
    main()
