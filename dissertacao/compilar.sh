#!/usr/bin/env bash
# Compila a dissertação localmente (~4s). Requer ~/.local/bin/tectonic
# (instalado em 05/10/2026; binário único, sem TeX Live).
cd "$(dirname "$0")"
tectonic main.tex && echo "OK -> $(pwd)/main.pdf"
