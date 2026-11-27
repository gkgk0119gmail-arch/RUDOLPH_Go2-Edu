#!/usr/bin/env bash
# NX 에서 녹화한 영상(제어판 '영상 녹화')을 이 PC 의 cv/data/baseline 으로 가져온다 (이미 있는 건 건너뜀).
#   cv/fetch.sh                    # 유선 unitree@192.168.123.18
#   RUDOLPH_NX=unitree@<wifi IP> cv/fetch.sh
set -eu
NX="${RUDOLPH_NX:-unitree@192.168.123.18}"
DST="$(cd "$(dirname "$0")" && pwd)/data/baseline"
mkdir -p "$DST"
rsync -av --ignore-existing --include='*.mp4' --exclude='*' "$NX:Desktop/3DOG/cv/data/baseline/" "$DST/"
ls "$DST"/*.mp4 2>/dev/null | wc -l | xargs echo "영상 개수:"
