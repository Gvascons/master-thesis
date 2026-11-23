"""Correção sequencial da manchete da QP3 (desenho em dois estágios).

Cronologia registrada no git: o pré-registro de 15/07 (eb0f223) fixa H1-H3
em 5 conjuntos e prevê extensão "+8-10"; em 20/07 às 11:34 (aa75dbe) o
pacote intermediário fecha com 11/15 positivas (sinal p=0,059) e o teste de
sinal aparece como instrumento; a regra de elegibilidade 3k-100k e os 20
conjuntos consolidam-se no mesmo dia. Logo a manchete 16/20 (p ingênuo
0,0059) foi lida após uma leitura intermediária, e o relato confirmatório
deve pagar o imposto sequencial correspondente (adendo de 23/11/2026 ao
desenho experimental; achado da auditoria externa de 23/11).

Cálculo exato sob H0 (p=0,5), duas leituras a 5% nominais:
  - críticos: X15 >= 12 (p=0,0176) e X20 >= 15 (p=0,0207)
  - erro tipo I global do desenho: 0,0283
  - p ajustado por ordenação de estágios para 16/20: 0,0189
  - extensão isolada (5/5 positivas): p = 0,03125
  - limite superior de Bonferroni (2 leituras): 2 x 0,0059 = 0,0118
"""
from scipy.stats import binom

p15 = lambda k: binom.sf(k - 1, 15, .5)
p20 = lambda k: binom.sf(k - 1, 20, .5)
c1 = min(k for k in range(16) if p15(k) <= .05)
c2 = min(k for k in range(21) if p20(k) <= .05)
print(f"criticos: X15>={c1} (p={p15(c1):.4f}), X20>={c2} (p={p20(c2):.4f})")

alpha = sum(binom.pmf(x, 15, .5) * binom.pmf(a, 5, .5)
            for x in range(16) for a in range(6) if x >= c1 or x + a >= c2)
print(f"alpha global (2 leituras a 5% nominais): {alpha:.4f}")

# p ajustado (ordenacao stage-wise): P(parar no 1o) + P(continuar e X20>=16)
padj = p15(c1) + sum(binom.pmf(x, 15, .5) * binom.sf(16 - x - 1, 5, .5)
                     for x in range(c1))
print(f"p ajustado stage-wise para 16/20: {padj:.4f}")
print(f"extensao isolada (5/5): {0.5**5:.5f}")
print(f"p ingenuo 16/20: {p20(16):.5f} | p interim 11/15: {p15(11):.5f}")
print(f"Bonferroni x2: {2*p20(16):.4f}")
