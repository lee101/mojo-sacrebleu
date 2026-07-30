"""SacreBLEU-compatible score and signature objects."""

from __future__ import annotations

import json
import statistics

from .version import __version__


class Score:
    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score
        self._mean = -1.0
        self._ci = -1.0
        self._verbose = ""

    def format(
        self,
        width: int = 2,
        score_only: bool = False,
        signature: str = "",
        is_json: bool = False,
    ) -> str:
        sc = f"{self.score:.{width}f}"
        data = {"name": self.name, "score": float(sc), "signature": signature}
        if self._mean > 0:
            mean = f"{self._mean:.{width}f}"
            interval = f"{self._ci:.{width}f}"
            confidence = f"μ = {mean} ± {interval}"
            sc += f" ({confidence})"
            data.update(
                confidence_mean=float(mean),
                confidence_var=float(interval),
                confidence=confidence,
            )
        if score_only:
            return sc
        full = f"{self.name}|{signature}" if signature else self.name
        full = f"{full} = {sc}"
        if self._verbose:
            full += f" {self._verbose}"
            data["verbose_score"] = self._verbose
        if is_json:
            for item in signature.split("|"):
                if item:
                    key, value = item.split(":", 1)
                    data[key] = value
            return json.dumps(data, indent=1, ensure_ascii=False)
        return full

    def estimate_ci(self, scores: list["Score"]) -> None:
        values = sorted(item.score for item in scores)
        lower_idx = len(values) // 40
        lower, upper = values[lower_idx], values[-lower_idx - 1]
        self._ci = 0.5 * (upper - lower)
        self._mean = statistics.mean(values)

    def __repr__(self) -> str:
        return self.format()


class Signature:
    abbreviations = {
        "version": "v",
        "nrefs": "#",
        "bs": "bs",
        "seed": "rs",
    }

    def __init__(self, info: dict):
        if not hasattr(info["metric"], "num_refs"):
            raise ValueError("Number of references unknown, please evaluate the metric first.")
        self.info = {
            "nrefs": "var" if info["metric"].num_refs == -1 else info["metric"].num_refs,
            "bs": info["metric"].n_bootstrap,
            "seed": info["metric"].seed,
            **info["values"],
            "version": __version__,
        }
        self._abbr = {**self.abbreviations, **info.get("abbreviations", {})}

    def format(self, short: bool = False) -> str:
        pairs = []
        keys = [key for key in self.info if key != "version"] + ["version"]
        for key in keys:
            value = self.info[key]
            if value is None:
                continue
            if isinstance(value, bool):
                value = "yes" if value else "no"
            pairs.append(f"{self._abbr.get(key, key) if short else key}:{value}")
        return "|".join(pairs)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()
