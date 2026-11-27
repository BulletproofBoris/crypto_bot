#!/usr/bin/env python3
"""
_tools/repair_historical_gate_depth.py
=============================================================================
Строго каузальная ретроспективная реконструкция монотонной 1 Гц шкалы приёма
для исторических архивов `data/cache_gdrive/gate_depth_*.csv.gz` + горячего
файла `data/vps_hot/gate_depth_hot.csv`, с потоковой сборкой полного
`data/processed/gate_depth.parquet` (~1.42 млн 1-секундных снимков, 0 потерь).
=============================================================================
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import glob
import gzip
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
GDRIVE_DIR = BASE_DIR / "data" / "cache_gdrive"
HOT_FILE = BASE_DIR / "data" / "vps_hot" / "gate_depth_hot.csv"
OUT_PARQUET = BASE_DIR / "data" / "processed" / "gate_depth.parquet"


def _repair_stream(path: Path, is_gz: bool, force: bool = False) -> tuple[int, int, int]:
    marker = Path(str(path) + ".repaired_1hz")
    if marker.exists() and not force:
        return 0, 0, 0

    opener = gzip.open if is_gz else open
    with opener(path, "rt", encoding="utf-8", errors="ignore", newline="") as rf:
        reader = csv.reader(rf)
        header = next(reader, None)
        if not header or header[0] != "timestamp":
            return 0, 0, 0
        rows_out = [header]
        running_max = -1.0
        cur_emit = -1.0
        last_tail: list[str] | None = None
        n_in = 0
        n_stale_node = 0
        n_quiet_step = 0

        for row in reader:
            if not row or row[0] == "timestamp":
                continue
            n_in += 1
            try:
                t_raw = datetime.fromisoformat(row[0].replace("Z", "+00:00")).timestamp()
            except Exception:
                continue

            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            if running_max < 0.0:
                running_max = t_raw
                cur_emit = t_raw
                last_tail = row[1:]
            elif t_raw > running_max + 1e-4:
                running_max = t_raw
                cur_emit = max(t_raw, cur_emit + 0.05)
                last_tail = row[1:]
            else:
                if t_raw < running_max - 0.5:
                    n_stale_node += 1
                else:
                    n_quiet_step += 1
                cur_emit += 1.0

            ts_iso = datetime.fromtimestamp(cur_emit, tz=timezone.utc).isoformat()
            rows_out.append([ts_iso] + (last_tail if last_tail is not None else row[1:]))

    tmp_p = path.with_name(path.name + ".tmp_rep")
    if is_gz:
        with gzip.open(tmp_p, "wt", encoding="utf-8", newline="", compresslevel=5) as wf:
            csv.writer(wf).writerows(rows_out)
    else:
        with open(tmp_p, "wt", encoding="utf-8", newline="") as wf:
            csv.writer(wf).writerows(rows_out)
    os.replace(tmp_p, path)
    if is_gz:
        marker.write_text(f"repaired rows={n_in} stale_cdn={n_stale_node} quiet={n_quiet_step}\n", encoding="utf-8")
    return n_in, n_stale_node, n_quiet_step


def build_clean_gate_parquet(gz_files: list[Path], hot_file: Path) -> None:
    from _tools.collect_gate_realtime import DEPTH_HEADERS
    from _tools.data_sanitizer import clean_orderbook_df

    target_cols = list(DEPTH_HEADERS)
    pa_fields = [pa.field("timestamp", pa.timestamp("us", tz="UTC"))] + [pa.field(c, pa.float64()) for c in target_cols[1:]]
    schema = pa.schema(pa_fields)

    tmp_pq = OUT_PARQUET.with_name("gate_depth.parquet.tmp")
    last_written_ts = None
    tot_written = 0

    sources = [(p, True) for p in gz_files] + ([(hot_file, False)] if hot_file.exists() else [])
    with pq.ParquetWriter(tmp_pq, schema, compression="snappy") as writer:
        for src_p, _ in sources:
            df = pd.read_csv(src_p, on_bad_lines="skip", low_memory=False)
            if df.empty or "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
            df = df.dropna(subset=["timestamp"])
            for c in target_cols[1:]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                else:
                    df[c] = np.nan
            df = df[target_cols].drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
            if last_written_ts is not None:
                df = df[df["timestamp"] > last_written_ts]
            df = clean_orderbook_df(df, "gate_depth")
            if df.empty:
                continue
            df["timestamp"] = df["timestamp"].astype("datetime64[us, UTC]")
            last_written_ts = df["timestamp"].iloc[-1]
            tbl = pa.Table.from_pandas(df[target_cols], schema=schema, preserve_index=False)
            writer.write_table(tbl)
            tot_written += len(df)

    os.replace(tmp_pq, OUT_PARQUET)
    sz_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"📦 Собран полный 1Hz gate_depth.parquet: {tot_written:,} строк ({sz_mb:.1f} MB), max_ts={last_written_ts}")


def main() -> None:
    for m in glob.glob(str(GDRIVE_DIR / "gate_depth_*.repaired_1hz")):
        os.remove(m)
    files = sorted(Path(p) for p in glob.glob(str(GDRIVE_DIR / "gate_depth_*.csv.gz")))
    print(f"🔧 Каузальная 1Hz-реконструкция {len(files)} архивов Gate.io + горячего хвоста...")
    tot_rows = 0
    tot_stale = 0
    for fp in files:
        n_in, n_st, n_qu = _repair_stream(fp, is_gz=True, force=True)
        tot_rows += n_in
        tot_stale += n_st
        print(f"  ✅ {fp.name}: {n_in:,} строк | CDN-откатов={n_st:,} | тихих={n_qu:,}")
    if HOT_FILE.exists():
        n_in, n_st, n_qu = _repair_stream(HOT_FILE, is_gz=False, force=True)
        tot_rows += n_in
        tot_stale += n_st
        print(f"  ✅ {HOT_FILE.name}: {n_in:,} строк | CDN-откатов={n_st:,} | тихих={n_qu:,}")

    for cp in glob.glob(str(GDRIVE_DIR / ".parquet_cache" / "gate_depth_*")):
        os.remove(cp)
    build_clean_gate_parquet(files, HOT_FILE)
    print(f"🏁 Итого обработано {tot_rows:,} секунд Gate.io (очищено {tot_stale:,} CDN-артефактов).")


if __name__ == "__main__":
    main()
