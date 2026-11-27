#!/usr/bin/env bash
# 이 컴퓨터에 맞는 zenoh-bridge-ros2dds 1.7.2 를 tools/zenoh/<arch>/ 로 받는다 (노트북·NX 둘 다, 한 번만).
#   ★ 1.7.2 로 고정: 1.10.2 x86_64 는 GLIBC 2.38 을 요구해 Ubuntu 22.04(2.35)에서 안 뜬다.
#     1.7.2 는 x86_64 GLIBC 2.34 · aarch64 GLIBC 2.29 까지만 요구 (노트북 22.04, NX 20.04 둘 다 됨). 양쪽 버전을 같게.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=1.7.2
A="$(uname -m)"
ZB="$HERE/$A/zenoh-bridge-ros2dds"
[ -x "$ZB" ] && { echo "이미 있음: $ZB ($("$ZB" --version 2>/dev/null | head -1))"; exit 0; }
mkdir -p "$HERE/$A"
T="$(mktemp -d)"
curl -fL -o "$T/zb.zip" "https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/$V/zenoh-plugin-ros2dds-$V-$A-unknown-linux-gnu-standalone.zip"
unzip -qo "$T/zb.zip" -d "$T/x"
cp "$T/x/zenoh-bridge-ros2dds" "$ZB"
chmod +x "$ZB"
rm -rf "$T"
"$ZB" --version | head -1
