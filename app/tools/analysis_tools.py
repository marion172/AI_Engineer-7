"""
app/tools/analysis_tools.py
===========================
Herramientas matemáticas y analíticas para el Agente de Análisis.
Nos permiten calcular porcentajes de crecimiento, ratios comparativos,
estadísticas descriptivas y evaluar riesgos a partir del texto de los reportes.
"""

import math
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


@tool
def calculate_growth_and_ratios(
    current_value: float,
    previous_value: float,
    benchmark_value: Optional[float] = None,
    metric_label: str = "Valor",
) -> Dict[str, Any]:
    """
    Calcula el crecimiento porcentual, la diferencia neta y el ratio
    frente a un valor previo o frente a un competidor (benchmark).
    """
    if previous_value == 0:
        pct_change = 0.0
    else:
        pct_change = ((current_value - previous_value) / abs(previous_value)) * 100.0

    absolute_diff = current_value - previous_value

    ratio_to_benchmark = None
    if benchmark_value and benchmark_value != 0:
        ratio_to_benchmark = round(current_value / benchmark_value, 2)

    return {
        "metric_label": metric_label,
        "current_value": current_value,
        "previous_value": previous_value,
        "absolute_difference": round(absolute_diff, 4),
        "percentage_growth": round(pct_change, 2),
        "ratio_to_benchmark": ratio_to_benchmark,
        "trend": "Crecimiento" if pct_change > 0 else ("Contracción" if pct_change < 0 else "Estable"),
    }


@tool
def compute_statistical_summary(values: List[float], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calcula la media, mediana, varianza, desviación estándar, mínimo y máximo
    de una serie de números recolectados.
    """
    if not values:
        return {"error": "La lista de valores está vacía."}

    n = len(values)
    mean_val = sum(values) / n
    sorted_vals = sorted(values)

    # Mediana
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[(n // 2) - 1] + sorted_vals[n // 2]) / 2.0

    # Varianza y desviación estándar
    variance = sum((x - mean_val) ** 2 for x in values) / (n if n == 1 else n - 1)
    std_dev = math.sqrt(variance)

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    labeled_data = []
    if labels and len(labels) == len(values):
        labeled_data = [{"label": l, "value": v} for l, v in zip(labels, values)]

    return {
        "count": n,
        "mean": round(mean_val, 3),
        "median": round(median_val, 3),
        "standard_deviation": round(std_dev, 3),
        "variance": round(variance, 3),
        "min": min_val,
        "max": max_val,
        "range": round(range_val, 3),
        "labeled_distribution": labeled_data,
    }


@tool
def evaluate_market_sentiment_and_risks(text_corpus: str) -> Dict[str, Any]:
    """
    Revisa el texto de los balances o noticias para medir el sentimiento general
    (positivo, neutral o cauteloso) y detectar alertas de riesgo como costos o energía.
    """
    positive_indicators = [
        "record", "growth", "grew", "increase", "up", "exceeded", "strong",
        "ramp", "profit", "margin", "accelerat", "optimis", "gain", "high"
    ]
    risk_indicators = [
        "cost", "risk", "delay", "supply", "inflation", "loss", "decline",
        "power", "energy", "bottleneck", "debt", "uncertain", "slowdown"
    ]

    corpus_lower = text_corpus.lower()
    pos_hits = sum(corpus_lower.count(p) for p in positive_indicators)
    risk_hits = sum(corpus_lower.count(r) for r in risk_indicators)
    total_tokens = max(1, pos_hits + risk_hits)

    # Polaridad normalizada de -1.0 a +1.0
    polarity = (pos_hits - risk_hits) / total_tokens

    if polarity > 0.2:
        classification = "Fuertemente Alcista / Positivo (Bullish)"
    elif polarity < -0.2:
        classification = "Cauteloso / Riesgo Elevado (Bearish)"
    else:
        classification = "Equilibrado / Neutral"

    return {
        "positive_signal_count": pos_hits,
        "risk_signal_count": risk_hits,
        "polarity_score": round(polarity, 3),
        "sentiment_classification": classification,
        "identified_risk_alerts": [
            r for r in risk_indicators if r in corpus_lower
        ],
    }

