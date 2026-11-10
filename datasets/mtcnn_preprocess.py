import argparse
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

def crop_face(image, detector, return_score=False):
    w, h = image.size
    scale = min(1.0, 640 / max(w, h))
    small = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BOX) if scale < 1 else image
    boxes, probs = detector.detect(small)

    if boxes is None:
        return (None, None) if return_score else None

    index = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[index] / scale
    pad_x, pad_y = 0.3 * (x2 - x1), 0.3 * (y2 - y1)

    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(w, int(np.ceil(x2 + pad_x)))
    bottom = min(h, int(np.ceil(y2 + pad_y)))

    if right <= left or bottom <= top:
        return (None, None) if return_score else None

    face = image.crop((left, top, right, bottom))
    return (face, float(probs[index])) if return_score else face

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["CASIA", "MSU", "OULU", "RA"], required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    detector = MTCNN(keep_all=True, device="cpu")
    extensions = {".avi", ".mp4", ".mov", ".mkv"}

    for video in sorted(args.input.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in extensions:
            continue

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        indices = [total // 2]

        if args.dataset == "OULU":
            label = "real" if video.stem.split("_")[-1] == "1" else "attack"
            prefix = f"{video.parent.name}__{video.stem}"
        elif args.dataset == "CASIA":
            label = "real" if video.parent.name == "real" else "attack"
            prefix = f"{video.parent.parent.name}__{video.stem}"
        elif args.dataset == "RA":
            parts = video.relative_to(args.input).parts
            split_name = parts[0]
            label = "real" if "real" in parts else "attack"
            subtype = video.parent.name
            name = video.stem
            if subtype not in ("real", "attack"):
                name = f"{subtype}_{name}"
            prefix = f"{split_name}__{name}"
        else:
            label = "real" if video.parent.name == "real" else "attack"
            prefix = video.stem

        destination = args.output / args.dataset / "full" / label
        destination.mkdir(parents=True, exist_ok=True)

        for number, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                decoded_frames = []

                while True:
                    success, current_frame = cap.read()
                    if not success:
                        break
                    decoded_frames.append(current_frame)

                if not decoded_frames:
                    break

                frame = decoded_frames[len(decoded_frames) // 2]

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = crop_face(image, detector)

            if face is not None:
                face.save(destination / f"{prefix}_{number:04d}.png")

        cap.release()


if __name__ == "__main__":
    main()
