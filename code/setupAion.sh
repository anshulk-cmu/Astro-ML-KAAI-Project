#!/usr/bin/env bash
# Clone, install, and inspect the Polymathic AION package on the box.
set +e
PY=/opt/pytorch/bin/python
PIP=/opt/pytorch/bin/pip
cd ~
[ -d AION ] || git clone --depth 1 https://github.com/PolymathicAI/AION.git
echo "===== ls AION ====="; ls AION
echo "===== ls AION/aion ====="; ls AION/aion 2>/dev/null
echo "===== pyproject ====="; sed -n '1,60p' AION/pyproject.toml 2>/dev/null
echo "===== README ====="; sed -n '1,180p' AION/README.md
echo "===== examples/notebooks ====="; find AION \( -iname "*example*" -o -iname "*.ipynb" -o -iname "*tutorial*" \) | head
echo "===== API grep ====="; grep -rn "from_pretrained\|CodecManager\|def encode\|LegacySurveyImage\|class AION" AION --include=*.py | head -40
echo "===== install ====="; $PIP install -e ./AION 2>&1 | tail -12
echo "===== import test ====="; $PY -c "from aion import AION; print('AION import ok')" 2>&1 | tail -5
