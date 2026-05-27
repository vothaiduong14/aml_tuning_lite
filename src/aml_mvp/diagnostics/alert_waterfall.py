"""Alert volume waterfall diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import struct
import zlib

import pandas as pd

from aml_mvp.storage import read_dataframe


def build_alert_waterfall(config: dict[str, Any], root: Path, logger=None) -> dict[str, Any]:
    artifacts = dict(config.get("artifacts", {}))
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    rule_hits = read_dataframe(_resolve(root, artifacts["rule_hits_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))

    rows = [
        {"stage": "transactions", "count": int(len(transactions)), "labels": int(transactions["is_laundering"].sum()) if "is_laundering" in transactions else 0},
    ]
    for rule_id, group in rule_hits.groupby("rule_id", sort=True):
        deduped = group.drop_duplicates("transaction_id")
        rows.append(
            {
                "stage": str(rule_id),
                "count": int(deduped["transaction_id"].nunique()),
                "labels": int(deduped[_label_column(deduped)].sum()) if _label_column(deduped) in deduped else 0,
            }
        )
    rows.append(
        {
            "stage": "consolidated_alerts",
            "count": int(len(alerts)),
            "labels": int(alerts[_label_column(alerts)].sum()) if _label_column(alerts) in alerts else 0,
        }
    )
    waterfall = pd.DataFrame(rows)
    chart_path = _resolve(root, artifacts["waterfall_chart_path"])
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_waterfall(waterfall, chart_path)
    if logger:
        logger.info("Wrote alert waterfall path=%s rows=%s", chart_path, len(waterfall))
    return {"chart_path": chart_path, "waterfall": waterfall}


def _plot_waterfall(waterfall: pd.DataFrame, path: Path) -> None:
    width, height = 900, 360
    margin = 36
    image = bytearray([255, 255, 255] * width * height)
    counts = waterfall["count"].astype(float).tolist()
    max_count = max(counts) if counts else 1.0
    bar_width = max(8, int((width - 2 * margin) / max(len(counts), 1) * 0.65))
    step = (width - 2 * margin) / max(len(counts), 1)
    for index, count in enumerate(counts):
        bar_height = int((height - 2 * margin) * (count / max_count)) if max_count else 0
        x0 = int(margin + index * step + (step - bar_width) / 2)
        y0 = height - margin - bar_height
        _fill_rect(image, width, height, x0, y0, bar_width, bar_height, (47, 93, 140))
    path.write_bytes(_png_bytes(width, height, image))


def _fill_rect(image: bytearray, width: int, height: int, x0: int, y0: int, rect_width: int, rect_height: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(height, y0 + rect_height)):
        for x in range(max(0, x0), min(width, x0 + rect_width)):
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)


def _png_bytes(width: int, height: int, rgb: bytearray) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb[y * width * 3 : (y + 1) * width * 3]) for y in range(height))

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _label_column(df: pd.DataFrame) -> str:
    return "target" if "target" in df.columns else "is_laundering"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
