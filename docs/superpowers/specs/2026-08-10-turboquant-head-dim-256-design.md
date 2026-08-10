# TurboQuant für head_dim als Vielfaches von 128

**Datum:** 2026-08-10
**Status:** Design abgenommen
**Ziel:** Qwen3.5-9B (head_dim 256) mit `--cache-type-k turbo3/turbo4` betreiben

## Problem

`llama-kv-cache.cpp` bricht die Kontext-Initialisierung ab, sobald ein Modell
head_dim ≠ 128 hat:

```
llama_kv_cache: TurboQuant requires head_dim=128, got 256 (layer 3)
llama_init_from_model: failed to initialize the context: turbo types require head_dim=128
```

Qwen3.5 (`Qwen3_5ForConditionalGeneration`) verwendet head_dim 256 und fällt
damit heraus. Betroffen sind die 8 `full_attention`-Layer; die übrigen 24
`linear_attention`-Layer halten einen rekurrenten State in F32 und berühren den
KV-Cache nicht.

## Befund: der Datenpfad ist bereits chunk-basiert

Die Beschränkung liegt ausschließlich im Guard, nicht in der Verarbeitung.
Alle drei Ebenen arbeiten in 128-Element-Chunks über einen flachen Speicher-
bereich und kennen den Begriff „Head" gar nicht:

| Ebene | Datei | Verhalten |
|---|---|---|
| CPU-Referenz | `ggml/src/ggml-quants.c:612` | `for (offset = 0; offset < k; offset += TURBO_HEAD_DIM)` |
| GPU-Write | `ggml/src/ggml-cuda/set-rows.cu:444` | `GGML_ASSERT(ne00 % TURBO_HEAD_DIM_SR == 0)`, ein CUDA-Block je Chunk |
| GPU-Read | `ggml/src/ggml-cuda/convert.cu:731` | `num_chunks = k / TURBO_HEAD_DIM_GPU`, ein CUDA-Block je Chunk |

Alle drei prüfen bereits auf Teilbarkeit, nicht auf Gleichheit. Ein Modell mit
head_dim 256 durchläuft sie unverändert korrekt — es entstehen zwei Chunks pro
Head statt einem.

## Ansatz: 2×128-Chunking

Ein 256-dimensionaler Head wird als zwei unabhängige 128er-Hälften behandelt.
Jede Hälfte wird separat L2-normalisiert, per FWHT rotiert und gegen das
bestehende Codebook quantisiert.

```
Head (256 dim)
 ├─ Hälfte A (128) → norm_A → FWHT128 → Codebook → 4 Blöcke
 └─ Hälfte B (128) → norm_B → FWHT128 → Codebook → 4 Blöcke
```

### Warum die Codebooks gültig bleiben

Die Zentroide sind Lloyd-Max-optimal für die Beta((d−1)/2, (d−1)/2)-Verteilung,
die entsteht, wenn ein **Einheitsvektor in R^128** per FWHT rotiert wird. Nach
der separaten Normalisierung ist jede Hälfte genau das: ein Einheitsvektor in
R^128. Die Verteilungsannahme gilt also exakt weiter — im Gegensatz zu einer
FWHT über 256 Dimensionen, die Beta(127,5, 127,5) erzeugen würde und neue
Zentroide (um Faktor 1/√2 enger) erfordert hätte.

### Warum kein zusätzlicher Speicher entsteht

Die Norm wird ohnehin redundant in jedem 32er-Block abgelegt
(`ggml-quants.c:636`, „Store same norm in every block of this chunk"). Zwei
Halb-Normen pro 256er-Head belegen dieselben Bytes wie eine Norm pro 128er-Head.
Die Bitrate bleibt unverändert bei 3,5 bpw (turbo3) bzw. 4,5 bpw (turbo4).

### Warum Chunks nie Head-Grenzen überspannen

Heads liegen kontiguierlich in der Row (`ne00 = n_head_kv × head_dim`). Da
head_dim ein Vielfaches der Chunk-Größe 128 ist, fällt jede Chunk-Grenze mit
einer Head-Grenze oder einer Head-Mitte zusammen, nie über einen Head hinaus.

## Änderungen

| Datei | Änderung |
|---|---|
| `src/llama-kv-cache.cpp:136-149` | Guard `!= 128` → `% 128 != 0`, Fehlertext anpassen |
| `tests/test-turboquant.cpp` | head_dim parametrisieren, Fälle 256 und 384 ergänzen |
| `docs/turboquant.md` | Requirements/Limitations korrigieren, Benchmark-Zeilen für Qwen3.5 |

`ggml-quants.c`, `set-rows.cu` und `convert.cu` bleiben unverändert.

## Nicht im Umfang

- head_dim 80/96/112 (kein Vielfaches von 128) — der Guard lehnt weiterhin ab
- Neu berechnete Codebooks für echte FWHT über 256
- Fusionierter Flash-Attention-Kernel (die Pre-Dequantize-Strategie bleibt)

## Verifikation

1. **Unit-Tests** — `test-turboquant` mit head_dim 128, 256, 384: FWHT-Selbst-
   inversion, Roundtrip-MSE·d gegen die Paper-Bereiche, Bitpack-Determinismus.
   Erwartung: MSE·d bei 256 im selben Bereich wie bei 128.
2. **Modell lädt** — Qwen3.5-9B-Q8_0 mit `-ctk turbo3 -ctv turbo3` startet und
   erzeugt kohärenten Text.
3. **Perplexität** — Qwen3.5-9B: f16 vs. turbo4 vs. turbo3. Erwartung:
   Delta in der Größenordnung der dokumentierten Werte (turbo4 ≈ +0,01,
   turbo3 ≈ +0,05).
4. **Durchsatz** — `llama-bench` pp512/tg128 gegen f16-Baseline.
5. **VRAM** — gemessener KV-Verbrauch gegen die Rechnung
   (turbo3 bei head_dim 256: 7,3 KiB/Token über 8 Attention-Layer).
6. **Vision** — mmproj-Pfad funktioniert weiterhin mit aktivem turbo3.

## Risiko und Rückfallweg

Die zwei Halb-Normen pro Head sind ein realer Unterschied zum d=128-Fall. Sie
erfassen Varianzunterschiede zwischen Head-Hälften feiner, was die Qualität eher
verbessert; belegt ist das erst durch Schritt 3. Fällt die Perplexität deutlich
schlechter aus als erwartet, ist der Rückfallweg eine echte FWHT über 256 mit
neu berechneten Lloyd-Max-Codebooks für Beta(127,5, 127,5).
