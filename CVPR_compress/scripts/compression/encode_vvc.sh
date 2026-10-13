#!/bin/bash

set -e

SRC="$HOME/knuvi-bce/MOTIP/datasets/DanceTrack"
DST="$HOME/knuvi-bce/MOTIP/datasets_vvc_qp37/DanceTrack"

VVENC="$HOME/knuvi-bce/vvenc/bin/release-static/vvencapp"
VVDEC="$HOME/knuvi-bce/vvdec/bin/release-static/vvdecapp"

QP=37

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

    # PNG 출력이므로 imExt 수정
    sed -i 's/^imExt=.*/imExt=.png/' "$OUT_SEQ/seqinfo.ini"

    # FPS / width / height / frame count 읽기
    FPS=$(grep '^frameRate=' "$SEQ_PATH/seqinfo.ini" | cut -d'=' -f2 | tr -d '\r')
    WIDTH=$(grep '^imWidth=' "$SEQ_PATH/seqinfo.ini" | cut -d'=' -f2 | tr -d '\r')
    HEIGHT=$(grep '^imHeight=' "$SEQ_PATH/seqinfo.ini" | cut -d'=' -f2 | tr -d '\r')
    FRAMES=$(grep '^seqLength=' "$SEQ_PATH/seqinfo.ini" | cut -d'=' -f2 | tr -d '\r')

    echo "FPS    : $FPS"
    echo "Size   : ${WIDTH}x${HEIGHT}"
    echo "Frames : $FRAMES"
    echo "QP     : $QP"

    TMP_YUV="/tmp/${SEQ}_input.yuv"
    TMP_BIT="/tmp/${SEQ}_qp${QP}.266"
    TMP_DEC="/tmp/${SEQ}_decoded.yuv"

    # 1) JPG sequence -> raw YUV420
    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -framerate "$FPS" \
        -start_number 1 \
        -i "$SEQ_PATH/img1/%08d.jpg" \
        -pix_fmt yuv420p \
        -f rawvideo \
        "$TMP_YUV"

    # 2) YUV -> VVC bitstream
    "$VVENC" \
        -i "$TMP_YUV" \
        -s "${WIDTH}x${HEIGHT}" \
        -r "$FPS" \
        --frames "$FRAMES" \
        --format yuv420 \
        --qp "$QP" \
        -o "$TMP_BIT"

    # 3) VVC -> decoded YUV
    "$VVDEC" \
        -b "$TMP_BIT" \
        -o "$TMP_DEC"

    # 4) decoded 10-bit YUV -> PNG sequence
    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -f rawvideo \
        -pix_fmt yuv420p10le \
        -s "${WIDTH}x${HEIGHT}" \
        -r "$FPS" \
        -i "$TMP_DEC" \
        -frames:v "$FRAMES" \
        -pix_fmt rgb24 \
        -start_number 1 \
        "$OUT_IMG/%08d.png"

    # temporary files 삭제
    rm -f "$TMP_YUV" "$TMP_BIT" "$TMP_DEC"

    SRC_COUNT=$(find "$SEQ_PATH/img1" -maxdepth 1 -name "*.jpg" | wc -l)
    DST_COUNT=$(find "$OUT_IMG" -maxdepth 1 -name "*.png" | wc -l)

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
