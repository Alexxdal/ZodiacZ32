# dedup_score_analyze.py
"""
Deduplica i risultati del solver Z‑13 e li riordina per «qualità».

*   deduplica sul trio *(segment, h, step)* tenendo il punteggio migliore
*   conta le “parole” plausibili (sequenze di ≥3 lettere) nel plaintext
*   ordina per **num_words desc**, poi **score asc**
*   salva in CSV e stampa le prime 20 righe

Uso:
    python dedup_score_analyze.py z13_candidates.csv [-o out.csv]
"""

from __future__ import annotations

import argparse
import re
import csv
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Z]{3,}")  # sequenze di ≥3 lettere maiuscole


def parse_csv_line(line: str) -> Dict[str, Any]:
    """Parsa *una* riga del CSV «sciolto» (6 campi, l'ultimo può contenere
    virgole).  Usata perché il campo «key» non è quotato, quindi `pd.read_csv`
    fallisce."""
    # taglia al massimo in 5 virgole: score, plaintext, segment, h, step,
    #   tutto il resto = key
    parts = line.rstrip("\n").split(",", 5)
    if len(parts) != 6:
        raise ValueError(f"Malformata (campi={len(parts)}): {line[:80]}…")

    score, plaintext, segment, h, step, key = parts
    return {
        "score": float(score),
        "plaintext": plaintext,
        "segment": segment,
        "h": h,          # es.: "20x?"  (non lo tocchiamo qui)
        "step": step,    # es.: "(1, 2)"
        "key": key,
    }


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Legge il CSV (non standard) restituendo un DataFrame."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf8", errors="ignore") as fh:
        header = fh.readline()  # salta la prima linea (titoli)
        for n, line in enumerate(fh, start=2):
            if not line.strip():
                continue  # salta linee vuote
            try:
                rows.append(parse_csv_line(line))
            except ValueError as err:
                print(f"⚠️  Riga {n}: {err}")
                continue
    return pd.DataFrame(rows)


def count_words(text: str) -> int:
    """Conta quante ‘parole’ plausibili (>=3 lettere) ci sono."""
    return len(WORD_RE.findall(text))

# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # Converte in tipi utili
    df["score"] = df["score"].astype(float)

    # Chiave di deduplica
    key_cols = ["segment", "h", "step"]

    # Mantiene la riga con score minore
    idx = (
        df.sort_values("score")
          .groupby(key_cols, as_index=False)
          .first().index
    )
    return df.loc[idx].reset_index(drop=True)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["words"] = df["plaintext"].map(count_words)
    return df


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["words", "score"], ascending=[False, True])

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplica e ordina i segmenti Z‑13")
    ap.add_argument("csv", type=Path, help="z13_candidates.csv da analizzare")
    ap.add_argument("-o", "--output", type=Path, default="z13_candidates_dedup.csv",
                    help="file CSV di destinazione (default: %(default)s)")
    args = ap.parse_args()

    # carica
    df = load_dataframe(args.csv)
    if df.empty:
        raise SystemExit("Nessuna riga valida trovata – controlla il file d'ingresso.")

    # pipeline
    df = deduplicate(df)
    df = enrich(df)
    df = sort_df(df)

    # salva
    df.to_csv(args.output, index=False)
    print(f"✅ Salvato in {args.output} – righe: {len(df)}")

    # anteprima
    print("\nTop 20:")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
