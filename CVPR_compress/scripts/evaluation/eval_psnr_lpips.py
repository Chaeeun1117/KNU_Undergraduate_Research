import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import lpips

from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor


def compute_psnr(ref, test):
    mse = torch.mean((ref - test) ** 2).item()

    if mse == 0:
        return float("inf")

    return -10.0 * math.log10(mse)


def load_tensor(path, device):
    img = Image.open(path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)
    return x


def find_matching_frame(img_dir, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ref",
        default=str(
            Path.home()
            / "Undergraduate/knuvi-bce/MOTIP/datasets_subset/DanceTrack"
        ),
    )

    parser.add_argument(
        "--test",
        default=str(
            Path.home()
            / "Undergraduate/knuvi-bce/MOTIP/datasets_subset_hevc_qp37/DanceTrack"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=str(
            Path.home()
            / "Undergraduate/knuvi-bce/MOTIP/datasets_subset_hevc_qp37"
        ),
    )

    args = parser.parse_args()

    ref_root = Path(args.ref)
    test_root = Path(args.test)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "psnr_lpips_per_frame.csv"
    log_path = out_dir / "psnr_lpips.log"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DanceTrack PSNR / LPIPS Evaluation")
    print("=" * 60)
    print(f"Reference : {ref_root}")
    print(f"Test      : {test_root}")
    print(f"Device    : {device}")
    print(f"CSV       : {csv_path}")
    print(f"LOG       : {log_path}")
    print()

    print("Loading LPIPS (AlexNet)...")

    lpips_model = lpips.LPIPS(
        net="alex"
    ).to(device)

    lpips_model.eval()

    sequences = sorted(
        p
        for p in (test_root / "val").glob("dancetrack*")
        if p.is_dir()
    )

    if not sequences:
        raise RuntimeError(
            f"No sequences found: {test_root / 'val'}"
        )

    all_results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "sequence",
            "frame",
            "psnr",
            "lpips",
        ])

        for seq_idx, test_seq in enumerate(sequences, 1):
            seq_name = test_seq.name

            ref_img_dir = (
                ref_root
                / "val"
                / seq_name
                / "img1"
            )

            test_img_dir = (
                test_seq
                / "img1"
            )

            if not ref_img_dir.exists():
                print(
                    f"[SKIP] Reference missing: {ref_img_dir}"
                )
                continue

            test_frames = sorted(
                p
                for p in test_img_dir.iterdir()
                if p.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            )

            print()
            print(
                f"===== [{seq_idx}/{len(sequences)}] "
                f"{seq_name} | {len(test_frames)} frames ====="
            )

            seq_results = []

            for test_path in tqdm(
                test_frames,
                desc=seq_name
            ):
                ref_path = find_matching_frame(
                    ref_img_dir,
                    test_path.stem
                )

                if ref_path is None:
                    print(
                        f"[WARN] Missing reference: "
                        f"{test_path.name}"
                    )
                    continue

                ref = load_tensor(
                    ref_path,
                    device
                )

                test = load_tensor(
                    test_path,
                    device
                )

                if ref.shape != test.shape:
                    raise RuntimeError(
                        f"Resolution mismatch\n"
                        f"REF : {ref_path} {ref.shape}\n"
                        f"TEST: {test_path} {test.shape}"
                    )

                #
                # PSNR
                #
                psnr_value = compute_psnr(
                    ref,
                    test
                )

                #
                # LPIPS expects [-1, 1]
                #
                ref_lpips = (
                    ref * 2.0 - 1.0
                )

                test_lpips = (
                    test * 2.0 - 1.0
                )

                with torch.no_grad():
                    lpips_value = (
                        lpips_model(
                            ref_lpips,
                            test_lpips
                        )
                        .item()
                    )

                writer.writerow([
                    seq_name,
                    test_path.name,
                    f"{psnr_value:.6f}",
                    f"{lpips_value:.6f}",
                ])

                f.flush()

                result = {
                    "sequence": seq_name,
                    "psnr": psnr_value,
                    "lpips": lpips_value,
                }

                seq_results.append(result)
                all_results.append(result)

            if seq_results:
                seq_psnr = np.mean(
                    [x["psnr"] for x in seq_results]
                )

                seq_lpips = np.mean(
                    [x["lpips"] for x in seq_results]
                )

                print(
                    f"\n[{seq_name}] "
                    f"PSNR={seq_psnr:.4f} dB | "
                    f"LPIPS={seq_lpips:.4f}"
                )

    if not all_results:
        raise RuntimeError(
            "No valid image pairs were evaluated."
        )

    global_psnr = np.mean(
        [x["psnr"] for x in all_results]
    )

    global_lpips = np.mean(
        [x["lpips"] for x in all_results]
    )

    sequence_names = sorted(
        set(
            x["sequence"]
            for x in all_results
        )
    )

    lines = []

    lines.append("=" * 60)
    lines.append(
        "DanceTrack PSNR / LPIPS Summary"
    )
    lines.append("=" * 60)

    lines.append(
        f"Reference : {ref_root}"
    )

    lines.append(
        f"Test      : {test_root}"
    )

    lines.append(
        f"Frames    : {len(all_results)}"
    )

    lines.append("")

    for seq_name in sequence_names:
        r = [
            x
            for x in all_results
            if x["sequence"] == seq_name
        ]

        p = np.mean(
            [x["psnr"] for x in r]
        )

        l = np.mean(
            [x["lpips"] for x in r]
        )

        lines.append(
            f"{seq_name}: "
            f"PSNR={p:.4f} dB | "
            f"LPIPS={l:.4f}"
        )

    lines.append("")
    lines.append("-" * 60)

    lines.append(
        f"AVERAGE PSNR : {global_psnr:.4f} dB"
    )

    lines.append(
        f"AVERAGE LPIPS: {global_lpips:.4f}"
    )

    lines.append("-" * 60)

    summary = "\n".join(lines)

    log_path.write_text(
        summary + "\n"
    )

    print()
    print(summary)

    print()
    print(f"[Saved] {log_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
