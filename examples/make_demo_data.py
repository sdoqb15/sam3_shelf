"""Generate synthetic, redistributable schema examples; not a training benchmark."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def generate(output):
    output = Path(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        rows = []
        for i in range(count):
            image = Image.new("RGB", (160, 100), (180 + i * 6, 190, 200))
            draw = ImageDraw.Draw(image)
            instances = []
            objects = 1 if i % 2 else 2
            for j in range(objects):
                box = (15 + j * 65, 20, 60 + j * 65, 85)
                color = (45 + i * 20, 100 + j * 35, 150 + len(split))
                draw.rectangle(box, fill=color)
                mask = Image.new("L", image.size, 0)
                ImageDraw.Draw(mask).rectangle(box, fill=255)
                name = f"{split}_{i}_{j}.png"
                mask.save(output / "masks" / name)
                instances.append(dict(mask="masks/" + name))
            # Distinguish decoded RGBs across splits, not only filenames.
            draw.point((0, 0), fill=(len(split) * 20, i * 30, 15))
            name = f"{split}_{i}.png"
            image.save(output / "images" / name)
            rows.append(
                dict(
                    id=f"{split}/{i}",
                    group=f"synthetic-{split}-{i}",
                    image="images/" + name,
                    width=160,
                    height=100,
                    sampling_group="single" if objects == 1 else "multi",
                    instances=instances,
                )
            )
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/demo"))
    generate(p.parse_args().output)
