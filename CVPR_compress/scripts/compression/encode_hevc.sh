#!/bin/bash

set -e

SRC="$HOME/knuvi-bce/MOTIP/datasets/DanceTrack"
DST="$HOME/knuvi-bce/MOTIP/datasets_hevc_qp42/DanceTrack"

QP=42

mkdir -p "$DST/val"

cp "$SRC/val_seqmap.txt" "$DST/val_seqmap.txt"

for SEQ_PATH in "$SRC"/val/dancetrack*; do

    [ -d "$SEQ_PATH" ] || continue

    SEQ=$(basename "$SEQ_PATH")

    echo "======================================"
    echo "[Processing] $SEQ"
    echo "======================================"

    OUT_SEQ="$DST/val/$SEQ"
    OUT_IMG="$OUT_SEQ/img1"

    mkdir -p "$OUT_IMG"

    # GT 복사
    if [ -d "$SEQ_PATH/gt" ]; then
        cp -r "$SEQ_PATH/gt" "$OUT_SEQ/"
    fi

    # seqinfo 복사
    cp "$SEQ_PATH/seqinfo.ini" "$OUT_SEQ/seqinfo.ini"

    # 출력 이미지 확장자를 PNG로 수정
    sed -i 's/^imExt=.*/imExt=.png/' "$OUT_SEQ/seqinfo.ini"

    # FPS 읽기
    FPS=$(grep '^frameRate=' "$SEQ_PATH/seqinfo.ini" \
        | cut -d'=' -f2 \
        | tr -d '\r')

    echo "FPS: $FPS"
    echo "QP : $QP"

    TMP_VIDEO="/tmp/${SEQ}_hevc_qp${QP}.mkv"

    # JPG sequence -> HEVC
    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -framerate "$FPS" \
        -start_number 1 \
        -i "$SEQ_PATH/img1/%08d.jpg" \
        -c:v libx265 \
        -preset medium \
        -x265-params "qp=${QP}" \
        -pix_fmt yuv420p \
        "$TMP_VIDEO"

    # HEVC -> PNG sequence
    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -i "$TMP_VIDEO" \
        -vsync 0 \
        -start_number 1 \
        "$OUT_IMG/%08d.png"

    rm "$TMP_VIDEO"

    SRC_COUNT=$(find "$SEQ_PATH/img1" \
        -maxdepth 1 -name "*.jpg" | wc -l)

    DST_COUNT=$(find "$OUT_IMG" \
        -maxdepth 1 -name "*.png" | wc -l)

    echo "Original frames: $SRC_COUNT"
    echo "Decoded frames : $DST_COUNT"

    if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
        echo "[ERROR] Frame count mismatch: $SEQ"
        exit 1
    fi

done

echo
echo "Done."
echo "Output: $DST"
