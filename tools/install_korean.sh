#!/bin/bash
# 한글 입력(ibus-hangul) + 한글 폰트 설치 (Ubuntu 20.04 / Jetson NX)
set -e

sudo apt update
sudo apt install -y ibus-hangul fonts-nanum fonts-noto-cjk language-pack-ko

# 입력기를 ibus로 지정
im-config -n ibus

# 입력 소스에 영어(us) + 한국어(Hangul) 추가
gsettings set org.gnome.desktop.input-sources sources "[('xkb', 'us'), ('ibus', 'hangul')]"

# 한/영 전환 키: 오른쪽 Alt(한/영 키), Shift+Space
gsettings set org.freedesktop.ibus.engine.hangul switch-keys 'Hangul,Shift+space,Alt_R' 2>/dev/null || true
# 오른쪽 Alt를 한/영 키로 인식시키기
gsettings set org.gnome.desktop.input-sources xkb-options "['korean:ralt_hangul']"

ibus restart 2>/dev/null || ibus-daemon -drx

echo ""
echo "설치 완료! 로그아웃 후 다시 로그인(또는 재부팅)하세요."
