import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# BILATERAL PAIRS
BILATERAL_PAIRS = {
    "outer_canthus": (33, 263),
    "inner_canthus": (133, 362),
    "upper_eyelid": (159, 386),
    "lower_eyelid": (145, 374),
    "eyebrow_inner": (55, 285),
    "eyebrow_peak": (52, 282),
    "eyebrow_outer": (46, 276),
    "nose_ala": (129, 358),
    "nose_wing": (64, 294),
    "mouth_corner": (61, 291),
    "upper_lip": (39, 269),
    "lower_lip": (91, 321),
    "cheekbone": (234, 454),
    "mid_cheek": (117, 346),
    "jaw_angle": (172, 397),
}

MIDLINE_IDX = {
    "forehead": 10,
    "nasion": 168,
    "nose_tip": 4,
    "subnasale": 1,
    "philtrum": 17,
    "chin": 152,
}


# DETECTOR
class FaceLandmarkDetector:
    def __init__(self, model_path):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect_bbox(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            faces = self.cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
        if len(faces) == 0:
            raise RuntimeError("No face detected")
        return max(faces, key=lambda f: f[2] * f[3])

    def run(self, image_bgr, bbox=None, side_factor=1.6):
        H, W = image_bgr.shape[:2]
        if bbox is None:
            bbox = self._detect_bbox(image_bgr)
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * side_factor
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(W, cx + side / 2))
        y1 = int(min(H, cy + side / 2))
        crop = image_bgr[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        net_in = cv2.resize(crop, (192, 192))
        net_in = cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interp.set_tensor(self.in_det["index"], net_in[None, ...])
        self.interp.invoke()
        lm = self.interp.get_tensor(self.out_det[0]["index"]).flatten().reshape(-1, 3)

        sx, sy = cw / 192.0, ch / 192.0
        xyz_crop = np.column_stack([lm[:, 0] * sx, lm[:, 1] * sy, lm[:, 2] * sx])
        xy_full = xyz_crop[:, :2].copy()
        xy_full[:, 0] += x0
        xy_full[:, 1] += y0
        return {
            "xyz_crop": xyz_crop,
            "xy_full": xy_full,
            "bbox_full": (x0, y0, x1, y1),
            "crop_size": (cw, ch),
        }

    def run_with_perturbations(
        self, image_path, n_replicates=5, perturb_frac=0.03, seed=0
    ):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        x, y, w, h = self._detect_bbox(image_bgr)
        rng = np.random.default_rng(seed)
        reps = [self.run(image_bgr, (x, y, w, h), side_factor=1.6)]
        for _ in range(n_replicates - 1):
            dx = rng.normal(0, perturb_frac * w)
            dy = rng.normal(0, perturb_frac * h)
            sf = rng.choice([1.5, 1.6, 1.7])
            reps.append(
                self.run(
                    image_bgr, (int(x + dx), int(y + dy), w, h), side_factor=float(sf)
                )
            )
        return reps, image_bgr


# POSE
def estimate_pose(xyz):
    lc = xyz[33, :2]
    rc = xyz[263, :2]
    nt = xyz[4, :2]
    dL = nt[0] - lc[0]
    dR = rc[0] - nt[0]
    ratio = (dL - dR) / (dL + dR + 1e-9)
    yaw = float(np.degrees(np.arcsin(np.clip(ratio, -1, 1))))
    roll = float(np.degrees(np.arctan2(rc[1] - lc[1], rc[0] - lc[0])))
    iod = float(np.linalg.norm(rc - lc))
    return {"yaw_deg": yaw, "roll_deg": roll, "iod_px": iod}


# TPS EXPORT — single-specimen streaming version
def write_tps_specimen(f, spec):
    coords = spec["coords"]
    n_dims = coords.shape[1]
    header = "LM" if n_dims == 2 else f"LM{n_dims}"
    f.write(f"{header}={coords.shape[0]}\n")
    for row in coords:
        f.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    if "image" in spec:
        f.write(f"IMAGE={spec['image']}\n")
    f.write(f"ID={spec['id']}\n")
    f.write(f"SCALE={spec.get('scale', 1.0)}\n")


# LAND.PAIRS EXPORT
def write_land_pairs(path, pairs):
    names = list(pairs.keys())
    K = len(names)
    with open(path, "w") as f:
        f.write(
            "pair_name,left_idx_1based,right_idx_1based,"
            "mediapipe_left,mediapipe_right\n"
        )
        for i, name in enumerate(names):
            mp_l, mp_r = pairs[name]
            f.write(f"{name},{i + 1},{K + i + 1},{mp_l},{mp_r}\n")


# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dossier photos (.jpg)")
    ap.add_argument("--output", required=True, help="Dossier de sortie")
    ap.add_argument("--model", required=True, help="Chemin face_landmark.tflite")
    ap.add_argument("--n-replicates", type=int, default=5)
    ap.add_argument("--perturb-frac", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(Path(args.input).rglob("*.jpg"))
        + list(Path(args.input).rglob("*.jpeg"))
        + list(Path(args.input).rglob("*.png"))
    )
    if not images:
        sys.exit(f"No images in {args.input}")

    detector = FaceLandmarkDetector(args.model)

    left_idx = [pairs[0] for pairs in BILATERAL_PAIRS.values()]
    right_idx = [pairs[1] for pairs in BILATERAL_PAIRS.values()]
    subset_idx = left_idx + right_idx
    n_landmarks = len(subset_idx)

    meta_fields = [
        "id", "photo_id", "replicate", "image_file",
        "yaw_deg", "roll_deg", "iod_px",
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "crop_w", "crop_h",
    ]

    n_specimens = 0

    # Open all output files before the loop — stream writes, no accumulation
    with (
        open(out_dir / "landmarks_paired.tps", "w") as tps_paired_f,
        open(out_dir / "landmarks_full.tps", "w") as tps_full_f,
        open(out_dir / "metadata.csv", "w", newline="") as meta_f,
        open(out_dir / "landmarks_fullimg.csv", "w", newline="") as fullimg_f,
    ):
        meta_writer = csv.DictWriter(meta_f, fieldnames=meta_fields)
        meta_writer.writeheader()

        fullimg_writer = csv.DictWriter(
            fullimg_f, fieldnames=["id", "landmark_idx", "x_px", "y_px"]
        )
        fullimg_writer.writeheader()

        for img_path in images:
            photo_id = img_path.stem
            print(f"Processing {photo_id}")
            try:
                reps, image_bgr = detector.run_with_perturbations(
                    img_path,
                    n_replicates=args.n_replicates,
                    perturb_frac=args.perturb_frac,
                    seed=hash(photo_id) & 0xFFFF,
                )
            except (RuntimeError, FileNotFoundError) as e:
                print(f"  /!\\ Skipping {photo_id}: {e}")
                continue

            for rep_idx, rep in enumerate(reps):
                rep_id = f"{photo_id}__rep{rep_idx:02d}"
                xyz_crop = rep["xyz_crop"]
                xy_full = rep["xy_full"]
                pose = estimate_pose(xyz_crop)

                # Write paired TPS specimen immediately
                write_tps_specimen(
                    tps_paired_f,
                    {
                        "id": rep_id,
                        "coords": xyz_crop[subset_idx, :2],
                        "image": img_path.name,
                        "scale": 1.0,
                    },
                )

                # Write full TPS specimen immediately
                write_tps_specimen(
                    tps_full_f,
                    {
                        "id": rep_id,
                        "coords": xyz_crop[:, :2],
                        "image": img_path.name,
                        "scale": 1.0,
                    },
                )

                # Write metadata row immediately
                meta_writer.writerow(
                    {
                        "id": rep_id,
                        "photo_id": photo_id,
                        "replicate": rep_idx,
                        "image_file": img_path.name,
                        "yaw_deg": round(pose["yaw_deg"], 4),
                        "roll_deg": round(pose["roll_deg"], 4),
                        "iod_px": round(pose["iod_px"], 4),
                        "bbox_x0": rep["bbox_full"][0],
                        "bbox_y0": rep["bbox_full"][1],
                        "bbox_x1": rep["bbox_full"][2],
                        "bbox_y1": rep["bbox_full"][3],
                        "crop_w": rep["crop_size"][0],
                        "crop_h": rep["crop_size"][1],
                    }
                )

                # Write fullimg rows immediately — no list accumulation
                for lm_idx in range(xy_full.shape[0]):
                    fullimg_writer.writerow(
                        {
                            "id": rep_id,
                            "landmark_idx": lm_idx,
                            "x_px": round(float(xy_full[lm_idx, 0]), 3),
                            "y_px": round(float(xy_full[lm_idx, 1]), 3),
                        }
                    )

                n_specimens += 1

            # Explicitly release the image array after processing all replicates
            del image_bgr
            del reps

    print(
        f"  -> {out_dir / 'landmarks_paired.tps'}  "
        f"({n_specimens} specimens × {n_landmarks} landmarks)"
    )
    print(
        f"  -> {out_dir / 'landmarks_full.tps'}  "
        f"({n_specimens} specimens × 468 landmarks)"
    )

    write_land_pairs(out_dir / "land_pairs.csv", BILATERAL_PAIRS)
    print(f"  -> {out_dir / 'land_pairs.csv'}")
    print(f"  -> {out_dir / 'metadata.csv'}")
    print(f"  -> {out_dir / 'landmarks_fullimg.csv'}")

    with open(out_dir / "bilateral_pairs.json", "w") as f:
        json.dump(
            {
                "pairs": BILATERAL_PAIRS,
                "midline": MIDLINE_IDX,
                "n_pairs": len(BILATERAL_PAIRS),
                "tps_format": "Export ordre : tous les L (1..K) puis tous les R (K+1..2K)",
            },
            f,
            indent=2,
        )
    print(f"  -> {out_dir / 'bilateral_pairs.json'}")

    print(
        f"\nDone. Processed {len(images)} images × {args.n_replicates} replicates "
        f"= {n_specimens} specimens."
    )


if __name__ == "__main__":
    main()
