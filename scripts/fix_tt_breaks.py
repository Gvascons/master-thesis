#!/usr/bin/env python3
"""Auditoria de layout (19/10/2026): insere pontos de quebra em caminhos
longos dentro de \\texttt{...} nas fontes LaTeX, eliminando overfull
hboxes de tokens inquebráveis (ex.: results/aggregated/test_results.csv
em legendas "Fonte:").

Regra: em todo argumento de \\texttt com comprimento > 14 contendo "/"
ou "\\_", insere "\\allowbreak" após cada "/" e após cada "\\_" (nunca
no último caractere). \\allowbreak é um penalty-0 — seguro em legendas,
células de tabela e argumentos móveis; não muda o conteúdo.

Idempotente (não duplica \\allowbreak). Uso:
  uv run python scripts/fix_tt_breaks.py arquivo1.tex [arquivo2.tex ...]
"""
import re
import sys
from pathlib import Path


def transform_arg(s: str) -> str:
    s = s.replace("\\allowbreak ", "")  # normaliza e reaplica (idempotente)
    if len(s) <= 14 or ("/" not in s and "\\_" not in s):
        return s
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] in "/-,." and i < n - 1:
            out.append(s[i] + "\\allowbreak ")
            i += 1
        elif s.startswith("\\_", i) and i + 2 < n:
            out.append("\\_\\allowbreak ")
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def process(text: str) -> tuple[str, int]:
    out = []
    i = 0
    count = 0
    key = "\\texttt{"
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        # varre o argumento com chaves balanceadas
        k = j + len(key)
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{" and text[k - 1] != "\\":
                depth += 1
            elif text[k] == "}" and text[k - 1] != "\\":
                depth -= 1
            k += 1
        arg = text[j + len(key):k - 1]
        new = transform_arg(arg)
        if new != arg:
            count += 1
        out.append(key + new + "}")
        i = k
    return "".join(out), count


def main():
    total = 0
    for f in sys.argv[1:]:
        p = Path(f)
        t = p.read_text()
        new, n = process(t)
        if n:
            p.write_text(new)
        print(f"{f}: {n} caminhos quebráveis")
        total += n
    print(f"total: {total}")


if __name__ == "__main__":
    main()
