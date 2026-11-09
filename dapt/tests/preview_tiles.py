import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("data", type=Path, help="Path to data cache directory")
    p.add_argument("out", type=Path, help="Path to output directory")
    p.add_argument("--start-index", type=int, default=0, help="First TRAIN index to preview; old numbering is retained")
    return p

def preview(args):
    cache = args.data
    output = args.out
    output.mkdir(parents=True, exist_ok=True)

    index_file = cache / "index.jsonl"
    records = [
        json.loads(line)
        for line in index_file.read_text().splitlines()
        if line.strip()
    ]

    # The diagnostic's --indices refer to positions within the TRAIN split,
    # not positions in the full index.jsonl file.
    train_records = [r for r in records if r["split"] == "train"]

    if not train_records:
        raise SystemExit("No training tiles found in " + str(index_file))

    if not 0 <= args.start_index < len(train_records):
        raise ValueError("start-index is outside TRAIN split")

    columns = 6
    rows = 6
    per_page = columns * rows

    for page_start in range(args.start_index, len(train_records), per_page):
        page_records = train_records[page_start:page_start + per_page]

        fig, axes = plt.subplots(
            rows, columns,
            figsize=(18, 19),
            constrained_layout=True,
        )

        for ax in axes.flat:
            ax.axis("off")

        for offset, record in enumerate(page_records):
            ax = axes.flat[offset]
            train_index = page_start + offset
            tile_path = cache / record["file"]

            with np.load(tile_path, allow_pickle=False) as tile:
                # Channel 0 is HHHH for the cache used in our training.
                image = tile["db"][0].astype(np.float32)
                valid = tile["valid"].astype(bool)

            valid &= np.isfinite(image)

            if not valid.any():
                ax.set_title(f"#{train_index}: no valid pixels")
                continue

            # Per-tile contrast stretching helps reveal shapes.
            low, high = np.percentile(image[valid], [2, 98])
            if high <= low:
                high = low + 1

            shown = np.ma.masked_where(~valid, image)
            ax.imshow(shown, cmap="gray", vmin=low, vmax=high)
            ax.set_title(
                f"train #{train_index}\n{record.get('area', 'unknown')}",
                fontsize=8,
            )

        page = page_start // per_page + 1
        fig.suptitle(f"Training tiles — page {page}", fontsize=16)

        destination = output / f"train_page_{page:04d}.png"
        fig.savefig(destination, dpi=120)
        plt.close(fig)

        print(f"Saved {destination}")

    print(f"\nDone. {len(train_records)} training tiles.")
    print(f"Open the PNG files in: {output.resolve()}")

def main():
    args = parser().parse_args()
    preview(args)


if __name__ == "__main__":
    main()
