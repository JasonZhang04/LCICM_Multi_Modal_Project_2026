"""
Extract aortic dilation labels from free-text echocardiogram reports.

Strategies:
1. Rule-based keyword matching (fast, interpretable)
2. Regex-based numeric extraction (aortic root diameter)
3. (Future) LLM-based extraction for ambiguous cases

Usage:
    from data.label_extractor import AorticDilationLabeler
    labeler = AorticDilationLabeler(config)
    label = labeler.extract_label(report_text)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AorticDilationLabeler:
    """Extract aortic dilation labels from echo report text."""

    # Severity grading keywords
    SEVERITY_PATTERNS = {
        "severe": [
            r"sever(?:e|ely)\s+(?:dilat|enlarg)",
            r"aneurysm(?:al)?\s+(?:aort|dilat)",
            r"aort\w*\s+aneurysm",
        ],
        "moderate": [
            r"moderat(?:e|ely)\s+(?:dilat|enlarg)",
        ],
        "mild": [
            r"mild(?:ly)?\s+(?:dilat|enlarg)",
            r"borderline\s+(?:dilat|enlarg)",
            r"slight(?:ly)?\s+(?:dilat|enlarg)",
        ],
    }

    # Regex for extracting aortic root diameter measurements
    DIAMETER_PATTERN = re.compile(
        r"aort\w*\s+root\s*(?:diameter|dimension|size)?\s*"
        r"(?:is|of|measures?|measured|=|:)?\s*"
        r"(\d+\.?\d*)\s*(cm|mm)",
        re.IGNORECASE,
    )

    # Alternative: just a number near "aortic root" context
    DIAMETER_CONTEXT_PATTERN = re.compile(
        r"aort\w*\s+root[^.]{0,40}?(\d+\.?\d*)\s*(cm|mm)",
        re.IGNORECASE,
    )

    def __init__(self, config: dict):
        """
        Args:
            config: The `labels` section of the YAML config.
        """
        self.mode = config.get("mode", "binary")
        self.positive_kw = [kw.lower() for kw in config.get("positive_keywords", [])]
        self.negative_kw = [kw.lower() for kw in config.get("negative_keywords", [])]
        self.diameter_threshold = config.get("diameter_threshold_cm", 4.0)

    def extract_label(self, report_text: str) -> dict:
        """
        Extract aortic dilation label from a single echo report.

        Returns:
            {
                "label": str or int,         # e.g., "dilated", "normal", 0, 1
                "confidence": str,           # "high", "medium", "low"
                "evidence": str,             # substring that triggered the label
                "diameter_cm": float | None, # extracted measurement if found
                "severity": str | None,      # "mild", "moderate", "severe" if detected
            }
        """
        text_lower = report_text.lower()
        result = {
            "label": None,
            "confidence": "low",
            "evidence": "",
            "diameter_cm": None,
            "severity": None,
        }

        # --- Step 1: Check explicit negative keywords first ---
        for kw in self.negative_kw:
            if kw in text_lower:
                result["label"] = self._format_label(False)
                result["confidence"] = "high"
                result["evidence"] = kw
                return result

        # --- Step 2: Try to extract numeric diameter ---
        diameter = self._extract_diameter(report_text)
        if diameter is not None:
            result["diameter_cm"] = diameter
            is_dilated = diameter >= self.diameter_threshold
            result["label"] = self._format_label(is_dilated)
            result["confidence"] = "high"
            result["evidence"] = f"aortic root diameter = {diameter:.1f} cm"
            if is_dilated:
                result["severity"] = self._severity_from_diameter(diameter)
            return result

        # --- Step 3: Check severity patterns ---
        for severity, patterns in self.SEVERITY_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Make sure it's in aortic context
                    context_start = max(0, match.start() - 50)
                    context = text_lower[context_start : match.end() + 50]
                    if "aort" in context:
                        result["label"] = self._format_label(True)
                        result["confidence"] = "high"
                        result["severity"] = severity
                        result["evidence"] = match.group()
                        return result

        # --- Step 4: Check positive keywords ---
        for kw in self.positive_kw:
            if kw in text_lower:
                result["label"] = self._format_label(True)
                result["confidence"] = "medium"
                result["evidence"] = kw
                return result

        # --- Step 5: No evidence found ---
        # Default: if no mention of aortic dilation at all, assume normal
        # but with low confidence
        result["label"] = self._format_label(False)
        result["confidence"] = "low"
        result["evidence"] = "no mention of aortic dilation"
        return result

    def _extract_diameter(self, text: str) -> Optional[float]:
        """Try to extract aortic root diameter in cm."""
        for pattern in [self.DIAMETER_PATTERN, self.DIAMETER_CONTEXT_PATTERN]:
            match = pattern.search(text)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                if unit == "mm":
                    value /= 10.0  # convert to cm
                # Sanity check: aortic root is typically 2-6 cm
                if 1.0 <= value <= 10.0:
                    return value
        return None

    def _severity_from_diameter(self, diameter_cm: float) -> str:
        """Estimate severity from diameter measurement."""
        if diameter_cm >= 5.5:
            return "severe"
        elif diameter_cm >= 4.5:
            return "moderate"
        elif diameter_cm >= self.diameter_threshold:
            return "mild"
        return "normal"

    def _format_label(self, is_dilated: bool):
        """Format label based on mode setting."""
        if self.mode == "binary":
            return 1 if is_dilated else 0
        else:
            # For ordinal mode, return string (severity assigned separately)
            return "dilated" if is_dilated else "normal"

    def batch_extract(self, reports: list[dict]) -> list[dict]:
        """
        Extract labels for a batch of reports.

        Args:
            reports: List of dicts with at least {"accession": str, "report_text": str}

        Returns:
            List of dicts with label info added.
        """
        results = []
        for item in reports:
            label_info = self.extract_label(item.get("report_text", ""))
            label_info["accession"] = item.get("accession", "")
            results.append(label_info)

        # Log summary
        labels = [r["label"] for r in results]
        if self.mode == "binary":
            n_pos = sum(1 for l in labels if l == 1)
            n_neg = sum(1 for l in labels if l == 0)
            logger.info(
                f"Label extraction complete: {n_pos} dilated, {n_neg} normal, "
                f"{len(labels)} total"
            )
        return results