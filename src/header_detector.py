import pandas as pd
from typing import List

from .models import DatasetProfile, HeaderInfo
from .header_hypothesis import HeaderHypothesisEngine
from .confidence import ConfidenceReport


class HeaderDetector:
    """Layer 3 — Confidence-driven header detection via multiple hypotheses."""

    HEADER_KEYWORDS = {
        "id", "date", "amount", "score", "total", "quantity",
        "age", "name", "category", "status", "description",
        "price", "value", "type", "index", "guid", "sum",
        "customer", "transaction", "order", "payment", "location",
    }

    def __init__(self, raw_df: pd.DataFrame, profile: DatasetProfile):
        self.raw_df = raw_df
        self.profile = profile
        self.confidence_report = ConfidenceReport()

    def _calculate_row_confidence(self, row_idx: int) -> float:
        rp = self.profile.row_profiles[row_idx]
        row_data = self.raw_df.iloc[row_idx].dropna().astype(str).str.lower()

        if rp.non_null_count == 0:
            return 0.0

        signals: dict = {}

        if rp.text_ratio > 0.8:
            signals["text"] = 0.30
        elif rp.text_ratio > 0.5:
            signals["text"] = 0.15

        if rp.numeric_ratio < 0.1:
            signals["low_numeric"] = 0.20
        elif rp.numeric_ratio > 0.5:
            signals["low_numeric"] = -0.35

        if rp.unique_token_ratio > 0.9:
            signals["unique"] = 0.15

        keyword_matches = sum(
            1 for val in row_data if any(kw in val for kw in self.HEADER_KEYWORDS)
        )
        if keyword_matches > 0:
            signals["keywords"] = min(0.30, keyword_matches * 0.10)

        if len(row_data) > 0:
            avg_len = sum(len(x) for x in row_data) / len(row_data)
            if avg_len < 15:
                signals["short_tokens"] = 0.10
            if avg_len > 25:
                signals["short_tokens"] = signals.get("short_tokens", 0) - 0.10

        density = (
            rp.non_null_count / self.profile.total_columns
            if self.profile.total_columns > 0
            else 0
        )
        if density > 0.3:
            signals["density"] = 0.10

        if len(rp.signature) > 0:
            t_ratio = rp.signature.count("T") / len(rp.signature)
            if t_ratio > 0.6:
                signals["sig_text"] = 0.15
            if "D" in rp.signature and rp.signature.count("D") > 1:
                signals["sig_date"] = -0.25
            if "N" in rp.signature and rp.numeric_ratio > 0.4:
                signals["sig_numeric"] = -0.20

        from .header_boundary_detector import _matches_anchor_value

        row = self.raw_df.iloc[row_idx]
        anchor_hits = sum(
            1
            for i in range(len(row))
            if pd.notna(row.iloc[i]) and _matches_anchor_value(row.iloc[i])
        )
        if anchor_hits > 0 and rp.non_null_count > 0:
            if anchor_hits / rp.non_null_count >= 0.2:
                signals["anchor_penalty"] = -0.40

        raw = sum(signals.values())
        return max(0.0, min(100.0, raw * 100))

    def detect(self) -> HeaderInfo:
        if not self.profile.row_profiles:
            return HeaderInfo([], 0, 0, 0.0, data_start_row=0)

        confidences = []
        for i in range(self.profile.total_rows):
            conf = self._calculate_row_confidence(i)
            confidences.append(conf)
            self.profile.row_profiles[i].header_confidence = conf

        engine = HeaderHypothesisEngine(self.raw_df, self.profile)
        scored = engine.evaluate(confidences)
        header_info = scored.value
        header_info.hypothesis_name = scored.name
        header_info.hypothesis_confidence = scored.confidence

        self.confidence_report.add_layer("header", scored.confidence, scored.signals)
        self.confidence_report.add_layer(
            "profiling", self.profile.profile_confidence
        )

        return header_info
