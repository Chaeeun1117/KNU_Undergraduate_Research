import argparse
import csv
import math
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from compressai.zoo import bmshj2018_hyperprior
from compressai.ops import compute_padding


def count_bits(strings):
    total_bytes = 0

    for latent_group in strings:
        for bitstream in latent_group:
            total_bytes += len(bitstream)

    return total_bytes * 8


def compute_psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2).item()

    if mse == 0:
        return float("inf")

    return -10.0 * math.log10(mse)


def copy_metadata(src_seq, dst_seq):
    dst_seq.mkdir(parents=True, exist_ok=True)

    for item in src_seq.iterdir():

        # img1은 따로 생성하므로 복사하지 않음
        if item.name == "img1":
            continue

        dst = dst_seq / item.name

        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(item, dst)

        else:
            shutil.copy2(item, dst)

    # NIC output frame은 PNG
    seqinfo = dst_seq / "seqinfo.ini"

    if seqinfo.exists():
        text = seqinfo.read_text()

        text = re.sub(
            r"(?m)^imExt\s*=.*$",
            "imExt=.png",
            text
        )

        seqinfo.write_text(text)


def load_existing_stats(stats_path):
    """
    기존 CSV를 읽어서
    (sequence, frame) -> row
    형태로 저장.

    resume 시 이미 처리된 frame을 확인하고,
    최종 summary도 전체 실행 결과 기준으로 계산하기 위함.
    """

    rows = {}

    if not stats_path.exists():
        return rows

    with open(stats_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            if (
                row.get("sequence") is None
                or row.get("frame") is None
            ):
                continue

            key = (
                row["sequence"],
                row["frame"]
            )

            rows[key] = row

    return rows


def rewrite_stats_csv(stats_path, rows):
    """
    중간 재실행으로 duplicate가 생겼더라도
    최종적으로 unique frame만 CSV에 남김.
    """

    fieldnames = [
        "sequence",
        "frame",
        "width",
        "height",
        "bits",
        "actual_bpp",
        "psnr",
        "encode_sec",
        "decode_sec"
    ]

    sorted_rows = sorted(
        rows.values(),
        key=lambda r: (
            r["sequence"],
            r["frame"]
        )
    )

    with open(
        stats_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(sorted_rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        default=str(
            Path.home()
            / "knuvi-bce/MOTIP/datasets_subset/DanceTrack"
        )
    )

    parser.add_argument(
        "--dst",
        required=True
    )

    parser.add_argument(
        "--quality",
        type=int,
        required=True,
        choices=range(1, 9)
    )

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    src_val = src_root / "val"
    dst_val = dst_root / "val"

    if not src_val.exists():
        raise FileNotFoundError(
            f"Source val directory not found: {src_val}"
        )

    #
    # Actual entropy coding은 CPU 사용
    #
    device = torch.device("cpu")

    print("=" * 60)
    print("Actual NIC Compression")
    print("Model   : BMSHJ2018-Hyperprior")
    print(f"Quality : {args.quality}")
    print(f"Device  : {device}")
    print(f"SRC     : {src_root}")
    print(f"DST     : {dst_root}")
    print("=" * 60)

    #
    # Load pretrained codec
    #
    model = bmshj2018_hyperprior(
        quality=args.quality,
        metric="mse",
        pretrained=True,
        progress=True
    ).to(device)

    model.eval()

    # entropy coding tables
    model.update(force=True)

    dst_root.mkdir(
        parents=True,
        exist_ok=True
    )

    dst_val.mkdir(
        parents=True,
        exist_ok=True
    )

    #
    # val_seqmap 복사
    #
    seqmap = src_root / "val_seqmap.txt"

    if seqmap.exists():
        shutil.copy2(
            seqmap,
            dst_root / "val_seqmap.txt"
        )

    #
    # DanceTrack sequences
    #
    sequences = sorted(
        p
        for p in src_val.glob("dancetrack*")
        if p.is_dir()
    )

    if not sequences:
        raise RuntimeError(
            f"No DanceTrack sequences found in {src_val}"
        )

    #
    # Statistics file
    #
    stats_path = (
        dst_root
        / f"bmshj_q{args.quality}_actual_stats.csv"
    )

    #
    # 이전 코드에서 잘못 cheng 이름으로 저장했던 CSV가 있으면
    # 자동으로 새 이름으로 이동
    #
    legacy_stats_path = (
        dst_root
        / f"cheng_q{args.quality}_actual_stats.csv"
    )

    if (
        not stats_path.exists()
        and legacy_stats_path.exists()
    ):
        print(
            f"[INFO] Migrating legacy stats:\n"
            f"       {legacy_stats_path.name}\n"
            f"    -> {stats_path.name}"
        )

        legacy_stats_path.rename(
            stats_path
        )

    #
    # 기존 통계 불러오기
    #
    existing_rows = load_existing_stats(
        stats_path
    )

    print(
        f"[Resume] Existing stats: "
        f"{len(existing_rows)} frames"
    )

    #
    # append mode
    #
    csv_exists = stats_path.exists()

    csvfile = open(
        stats_path,
        "a",
        newline=""
    )

    fieldnames = [
        "sequence",
        "frame",
        "width",
        "height",
        "bits",
        "actual_bpp",
        "psnr",
        "encode_sec",
        "decode_sec"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    if not csv_exists:
        writer.writeheader()
        csvfile.flush()

    newly_processed = 0
    skipped = 0

    try:

        for seq_idx, src_seq in enumerate(
            sequences,
            1
        ):

            seq_name = src_seq.name

            print()
            print(
                f"===== [{seq_idx}/{len(sequences)}] "
                f"{seq_name} ====="
            )

            dst_seq = (
                dst_val
                / seq_name
            )

            copy_metadata(
                src_seq,
                dst_seq
            )

            src_img_dir = (
                src_seq
                / "img1"
            )

            dst_img_dir = (
                dst_seq
                / "img1"
            )

            #
            # 기존 결과 삭제하지 않음
            #
            dst_img_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            images = sorted(
                list(src_img_dir.glob("*.jpg"))
                + list(src_img_dir.glob("*.jpeg"))
                + list(src_img_dir.glob("*.png"))
            )

            if not images:
                print(
                    f"[WARN] No frames: {src_img_dir}"
                )
                continue

            for idx, img_path in enumerate(
                images,
                1
            ):

                out_path = (
                    dst_img_dir
                    / f"{img_path.stem}.png"
                )

                key = (
                    seq_name,
                    img_path.name
                )

                #
                # ★ 핵심 resume 부분
                #
                # output PNG도 존재하고
                # stats CSV에도 기록이 있으면
                # 실제 codec 실행 없이 바로 skip
                #
                if (
                    out_path.exists()
                    and key in existing_rows
                ):
                    skipped += 1

                    if (
                        idx == 1
                        or idx % 50 == 0
                        or idx == len(images)
                    ):
                        print(
                            f"[SKIP {idx:5d}/{len(images):5d}] "
                            f"{img_path.name}"
                        )

                    continue

                #
                # PNG는 있는데 stats가 없다면
                # crash가 PNG 저장 직후 발생했을 가능성이 있으므로
                # 해당 frame만 다시 실제 압축하여 stats도 복구
                #
                if (
                    out_path.exists()
                    and key not in existing_rows
                ):
                    print(
                        f"[RECOVER] {img_path.name}: "
                        f"output exists but stats missing"
                    )

                image = (
                    Image.open(img_path)
                    .convert("RGB")
                )

                x = (
                    to_tensor(image)
                    .unsqueeze(0)
                    .to(device)
                )

                _, _, h, w = x.shape

                #
                # CompressAI codec 사용을 위해
                # spatial dimension을 64 배수로 padding
                #
                pad, unpad = compute_padding(
                    h,
                    w,
                    min_div=64
                )

                x_pad = F.pad(
                    x,
                    pad,
                    mode="constant",
                    value=0
                )

                with torch.no_grad():

                    #
                    # Actual entropy encoding
                    #
                    start = time.time()

                    encoded = model.compress(
                        x_pad
                    )

                    encode_time = (
                        time.time()
                        - start
                    )

                    #
                    # Actual entropy decoding
                    #
                    start = time.time()

                    decoded = model.decompress(
                        encoded["strings"],
                        encoded["shape"]
                    )

                    decode_time = (
                        time.time()
                        - start
                    )

                    x_hat = decoded["x_hat"]

                    #
                    # original resolution로 crop
                    #
                    x_hat = F.pad(
                        x_hat,
                        unpad
                    )

                    x_hat = x_hat.clamp(
                        0,
                        1
                    )

                #
                # Actual bitstream size
                #
                bits = count_bits(
                    encoded["strings"]
                )

                actual_bpp = (
                    bits
                    / (h * w)
                )

                psnr = compute_psnr(
                    x,
                    x_hat
                )

                #
                # reconstructed frame 저장
                #
                to_pil_image(
                    x_hat.squeeze(0)
                ).save(
                    out_path
                )

                #
                # stats 저장
                #
                row = {
                    "sequence": seq_name,
                    "frame": img_path.name,
                    "width": w,
                    "height": h,
                    "bits": bits,
                    "actual_bpp": f"{actual_bpp:.8f}",
                    "psnr": f"{psnr:.4f}",
                    "encode_sec": f"{encode_time:.4f}",
                    "decode_sec": f"{decode_time:.4f}"
                }

                writer.writerow(row)

                #
                # crash에 대비해서 매 frame flush
                #
                csvfile.flush()

                existing_rows[key] = {
                    k: str(v)
                    for k, v in row.items()
                }

                newly_processed += 1

                if (
                    idx == 1
                    or idx % 10 == 0
                    or idx == len(images)
                ):

                    print(
                        f"[{idx:5d}/{len(images):5d}] "
                        f"{img_path.name} | "
                        f"bpp={actual_bpp:.4f} | "
                        f"PSNR={psnr:.2f} dB | "
                        f"enc={encode_time:.2f}s | "
                        f"dec={decode_time:.2f}s"
                    )

    finally:
        csvfile.close()

    #
    # CSV 중복 제거 / 정리
    #
    rewrite_stats_csv(
        stats_path,
        existing_rows
    )

    #
    # 최종 summary는 현재 run만이 아니라
    # CSV에 저장된 전체 frame 기준으로 계산
    #
    final_rows = load_existing_stats(
        stats_path
    )

    if not final_rows:
        raise RuntimeError(
            "No frames were processed or found."
        )

    total_bits = 0
    total_pixels = 0

    total_psnr = 0.0
    total_encode_time = 0.0
    total_decode_time = 0.0

    valid_frames = 0

    for row in final_rows.values():

        try:
            w = int(row["width"])
            h = int(row["height"])
            bits = int(row["bits"])

            psnr = float(
                row["psnr"]
            )

            enc = float(
                row["encode_sec"]
            )

            dec = float(
                row["decode_sec"]
            )

        except (
            ValueError,
            TypeError,
            KeyError
        ):
            continue

        total_bits += bits
        total_pixels += (
            w * h
        )

        total_psnr += psnr
        total_encode_time += enc
        total_decode_time += dec

        valid_frames += 1

    if valid_frames == 0:
        raise RuntimeError(
            "No valid statistics found."
        )

    avg_bpp = (
        total_bits
        / total_pixels
    )

    avg_psnr = (
        total_psnr
        / valid_frames
    )

    avg_enc = (
        total_encode_time
        / valid_frames
    )

    avg_dec = (
        total_decode_time
        / valid_frames
    )

    summary = (
        f"Model: BMSHJ2018-Hyperprior\n"
        f"Metric: MSE\n"
        f"Quality: {args.quality}\n"
        f"Entropy coding: actual ANS bitstream\n"
        f"Frames: {valid_frames}\n"
        f"Actual average bpp: {avg_bpp:.8f}\n"
        f"Average PSNR: {avg_psnr:.4f} dB\n"
        f"Average encode time: {avg_enc:.4f} sec/frame\n"
        f"Average decode time: {avg_dec:.4f} sec/frame\n"
        f"Newly processed this run: {newly_processed}\n"
        f"Skipped existing frames: {skipped}\n"
    )

    summary_path = (
        dst_root
        / f"bmshj_q{args.quality}_actual_summary.txt"
    )

    summary_path.write_text(
        summary
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(summary)

    print(
        f"Stats   : {stats_path}"
    )

    print(
        f"Summary : {summary_path}"
    )


if __name__ == "__main__":
    main()
