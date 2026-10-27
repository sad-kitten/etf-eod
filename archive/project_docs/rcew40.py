#!/usr/bin/env python3
"""
rcew40.py

Auditable RCEW40 calculator:
- 40-year robust certainty-equivalent terminal wealth
- net of portfolio weighted expense ratio
- with fixed 3% annual draw from initial corpus
- using log/geometric aggregation across scenarios

Default convention:
    Year-end draw:
        W_t = W_{t-1} * (1 + net_cagr) - annual_draw

Where:
    annual_draw = annual_draw_rate * initial_wealth

This means a 3% annual draw on $2.1M is $63,000/year, fixed in nominal terms
unless the caller changes annual_draw_rate or initial_wealth.

Important:
    This module DOES NOT invent stress weights.
    If you want the full robust blend:
        50% canonical + 25% equal + 25% stress
    provide explicit StressWeight values in your scenario table.

Scenario CSV expected columns by default:
    Scenario, Weight, GrossCAGR
Optional:
    StressWeight

Portfolio CSV expected columns by default:
    Ticker, Weight, ExpenseRatio

Rates may be supplied as decimals or percents:
    0.08 or 8 both parse as 8%
    0.0035 or 0.35 both parse as 0.35%

Example CLI:
    python rcew40.py scenarios.csv --portfolio portfolio.csv --initial-wealth 2100000

Example with direct portfolio ER:
    python rcew40.py scenarios.csv --portfolio-er 0.0042 --output results.json

Authoring note:
    This file is intentionally small, boring, and inspectable.
    No pandas. No hidden model. No oracle in the basement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Any


EPSILON_DEFAULT = 1e-12


@dataclass(frozen=True)
class RCEW40Config:
    horizon_years: int = 40
    annual_draw_rate: float = 0.03
    initial_wealth: float = 1.0
    draw_timing: str = "year_end"  # "year_end" or "year_begin"
    epsilon: float = EPSILON_DEFAULT
    robust_blend_canonical: float = 0.50
    robust_blend_equal: float = 0.25
    robust_blend_stress: float = 0.25
    require_stress_for_robust_blend: bool = True


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    canonical_weight: float
    gross_cagr: float
    net_cagr: float
    terminal_wealth_multiple: float
    terminal_wealth_dollars: float
    stress_weight: Optional[float] = None
    ruined: bool = False


@dataclass(frozen=True)
class RCEW40Result:
    config: Dict[str, Any]
    portfolio_expense_ratio: float
    rcew40_canonical: float
    rcew40_equal: float
    rcew40_stress: Optional[float]
    rcew40_robust_blend: Optional[float]
    robust_blend_available: bool
    notes: List[str]
    scenario_results: List[Dict[str, Any]]


def parse_rate(value: Any) -> float:
    """
    Parse a decimal/percent rate.

    Examples:
        "8%"     -> 0.08
        "8"      -> 0.08
        "0.08"   -> 0.08
        "0.35%"  -> 0.0035
        "0.35"   -> 0.0035 if intended as percent for ER? Ambiguous.

    Rule:
        - Values ending in % are divided by 100.
        - Numeric abs(value) > 1 is treated as percent and divided by 100.
        - Numeric abs(value) <= 1 is treated as decimal.

    For expense ratios, prefer explicit decimals such as 0.0042 for 0.42%.
    """
    if value is None:
        raise ValueError("Cannot parse None as rate.")
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        s = str(value).strip().replace(",", "")
        if s == "":
            raise ValueError("Cannot parse blank value as rate.")
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        x = float(s)
    if abs(x) > 1.0:
        return x / 100.0
    return x


def parse_weight(value: Any) -> float:
    """
    Parse a weight as decimal or percent using the same broad rule as parse_rate.
    """
    return parse_rate(value)


def normalize_weights(weights: Sequence[float]) -> List[float]:
    total = sum(float(w) for w in weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [float(w) / total for w in weights]


def weighted_expense_ratio(
    portfolio_rows: Sequence[Mapping[str, Any]],
    weight_col: str = "Weight",
    expense_ratio_col: str = "ExpenseRatio",
) -> float:
    """
    Calculate portfolio weighted expense ratio.

    portfolio_rows example:
        [
            {"Ticker": "SHLD", "Weight": 0.15, "ExpenseRatio": 0.0050},
            {"Ticker": "CME",  "Weight": 0.12, "ExpenseRatio": 0.0000},
        ]
    """
    weights = [parse_weight(row[weight_col]) for row in portfolio_rows]
    ers = [parse_rate(row[expense_ratio_col]) for row in portfolio_rows]
    w_norm = normalize_weights(weights)
    return sum(w * er for w, er in zip(w_norm, ers))


def terminal_wealth_multiple_after_fixed_draw(
    net_cagr: float,
    *,
    horizon_years: int = 40,
    annual_draw_rate: float = 0.03,
    draw_timing: str = "year_end",
) -> float:
    """
    Terminal wealth multiple after fixed annual draw from initial wealth.

    Uses normalized initial wealth W0 = 1.0, so annual_draw = annual_draw_rate.

    Year-end draw:
        W_t = W_{t-1} * (1 + r) - D

    Year-begin draw:
        W_t = (W_{t-1} - D) * (1 + r)

    Returns:
        W_horizon / W0
    """
    r = float(net_cagr)
    d = float(annual_draw_rate)

    if horizon_years < 0:
        raise ValueError("horizon_years must be non-negative.")
    if draw_timing not in {"year_end", "year_begin"}:
        raise ValueError("draw_timing must be 'year_end' or 'year_begin'.")

    wealth = 1.0
    for _ in range(int(horizon_years)):
        if draw_timing == "year_end":
            wealth = wealth * (1.0 + r) - d
        else:
            wealth = (wealth - d) * (1.0 + r)
    return wealth


def terminal_wealth_multiple_closed_form_year_end(
    net_cagr: float,
    *,
    horizon_years: int = 40,
    annual_draw_rate: float = 0.03,
) -> float:
    """
    Closed-form terminal wealth multiple for year-end draw.

    If r != 0:
        M = (1+r)^H - D * (((1+r)^H - 1) / r)

    If r == 0:
        M = 1 - D * H

    Kept mainly for audit/reference. The iterative function is easier to generalize.
    """
    r = float(net_cagr)
    h = int(horizon_years)
    d = float(annual_draw_rate)

    if abs(r) < 1e-15:
        return 1.0 - d * h

    growth = (1.0 + r) ** h
    return growth - d * ((growth - 1.0) / r)


def log_certainty_equivalent(
    terminal_multiples: Sequence[float],
    weights: Sequence[float],
    *,
    epsilon: float = EPSILON_DEFAULT,
) -> float:
    """
    Scenario-weighted geometric/log certainty-equivalent terminal wealth.

    CEW = exp(sum(weight_s * ln(max(M_s, epsilon))))

    If terminal wealth <= 0, the scenario is effectively ruin.
    The epsilon floor prevents math failure but should be treated as a red flag.
    """
    if len(terminal_multiples) != len(weights):
        raise ValueError("terminal_multiples and weights must have the same length.")
    w_norm = normalize_weights(weights)
    log_sum = 0.0
    for m, w in zip(terminal_multiples, w_norm):
        safe_m = max(float(m), epsilon)
        log_sum += w * math.log(safe_m)
    return math.exp(log_sum)


def robust_log_blend(
    canonical_cew: float,
    equal_cew: float,
    stress_cew: Optional[float],
    *,
    canonical_weight: float = 0.50,
    equal_weight: float = 0.25,
    stress_weight: float = 0.25,
    require_stress: bool = True,
    epsilon: float = EPSILON_DEFAULT,
) -> Optional[float]:
    """
    Blend CEW legs in log space.

    Standard RCEW blend:
        exp(0.50*ln(canonical) + 0.25*ln(equal) + 0.25*ln(stress))

    If stress_cew is missing and require_stress=True, returns None.
    If stress_cew is missing and require_stress=False, re-normalizes available legs.
    """
    legs: List[Tuple[float, float]] = [
        (canonical_weight, canonical_cew),
        (equal_weight, equal_cew),
    ]

    if stress_cew is None:
        if require_stress:
            return None
    else:
        legs.append((stress_weight, stress_cew))

    total_weight = sum(w for w, _ in legs)
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value.")

    log_sum = 0.0
    for w, value in legs:
        log_sum += (w / total_weight) * math.log(max(float(value), epsilon))
    return math.exp(log_sum)


def compute_rcew40(
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    portfolio_expense_ratio: float,
    scenario_col: str = "Scenario",
    canonical_weight_col: str = "Weight",
    gross_cagr_col: str = "GrossCAGR",
    stress_weight_col: Optional[str] = "StressWeight",
    config: RCEW40Config = RCEW40Config(),
) -> RCEW40Result:
    """
    Compute RCEW40 for a portfolio when scenario gross CAGRs are already mapped.

    scenario_rows example:
        [
            {"Scenario": "AI compute boom", "Weight": 0.0777, "GrossCAGR": 0.105, "StressWeight": 0.05},
            {"Scenario": "Credit accident", "Weight": 0.0583, "GrossCAGR": 0.042, "StressWeight": 0.10},
        ]

    portfolio_expense_ratio is subtracted from each scenario CAGR:
        net_cagr = gross_cagr - portfolio_expense_ratio
    """
    if not scenario_rows:
        raise ValueError("scenario_rows cannot be empty.")

    er = parse_rate(portfolio_expense_ratio)

    scenario_names: List[str] = []
    canonical_weights_raw: List[float] = []
    gross_cagrs: List[float] = []
    stress_weights_raw: List[float] = []
    stress_available = True

    for idx, row in enumerate(scenario_rows):
        scenario_names.append(str(row.get(scenario_col, f"Scenario {idx + 1}")))
        canonical_weights_raw.append(parse_weight(row[canonical_weight_col]))
        gross_cagrs.append(parse_rate(row[gross_cagr_col]))

        if stress_weight_col and stress_weight_col in row and str(row[stress_weight_col]).strip() != "":
            stress_weights_raw.append(parse_weight(row[stress_weight_col]))
        else:
            stress_available = False

    canonical_weights = normalize_weights(canonical_weights_raw)
    equal_weights = [1.0 / len(scenario_rows)] * len(scenario_rows)

    stress_weights: Optional[List[float]]
    if stress_available and len(stress_weights_raw) == len(scenario_rows):
        stress_weights = normalize_weights(stress_weights_raw)
    else:
        stress_weights = None

    net_cagrs = [g - er for g in gross_cagrs]
    terminal_multiples = [
        terminal_wealth_multiple_after_fixed_draw(
            r,
            horizon_years=config.horizon_years,
            annual_draw_rate=config.annual_draw_rate,
            draw_timing=config.draw_timing,
        )
        for r in net_cagrs
    ]

    rcew_canonical = log_certainty_equivalent(
        terminal_multiples, canonical_weights, epsilon=config.epsilon
    )
    rcew_equal = log_certainty_equivalent(
        terminal_multiples, equal_weights, epsilon=config.epsilon
    )
    rcew_stress = (
        log_certainty_equivalent(terminal_multiples, stress_weights, epsilon=config.epsilon)
        if stress_weights is not None
        else None
    )

    rcew_robust = robust_log_blend(
        rcew_canonical,
        rcew_equal,
        rcew_stress,
        canonical_weight=config.robust_blend_canonical,
        equal_weight=config.robust_blend_equal,
        stress_weight=config.robust_blend_stress,
        require_stress=config.require_stress_for_robust_blend,
        epsilon=config.epsilon,
    )

    scenario_results = []
    for i, name in enumerate(scenario_names):
        terminal_dollars = terminal_multiples[i] * config.initial_wealth
        scenario_results.append(
            ScenarioResult(
                scenario=name,
                canonical_weight=canonical_weights[i],
                gross_cagr=gross_cagrs[i],
                net_cagr=net_cagrs[i],
                terminal_wealth_multiple=terminal_multiples[i],
                terminal_wealth_dollars=terminal_dollars,
                stress_weight=stress_weights[i] if stress_weights is not None else None,
                ruined=terminal_multiples[i] <= 0,
            )
        )

    notes: List[str] = [
        "GrossCAGR is reduced by portfolio weighted expense ratio before terminal wealth is calculated.",
        f"Annual draw is fixed at {config.annual_draw_rate:.6f} of initial wealth per year.",
        f"Draw timing convention: {config.draw_timing}.",
        "Scenario aggregation uses log/geometric certainty-equivalent wealth, not arithmetic averaging.",
    ]

    if stress_weights is None:
        notes.append(
            "No complete StressWeight column was supplied; stress RCEW40 and full 50/25/25 robust blend are unavailable."
        )
    if any(sr.ruined for sr in scenario_results):
        notes.append(
            f"At least one scenario had terminal wealth <= 0. Log CEW used epsilon={config.epsilon} for computation."
        )

    return RCEW40Result(
        config=asdict(config),
        portfolio_expense_ratio=er,
        rcew40_canonical=rcew_canonical,
        rcew40_equal=rcew_equal,
        rcew40_stress=rcew_stress,
        rcew40_robust_blend=rcew_robust,
        robust_blend_available=rcew_robust is not None,
        notes=notes,
        scenario_results=[asdict(sr) for sr in scenario_results],
    )


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate RCEW40 net of expense ratio and fixed annual draw."
    )
    parser.add_argument("scenarios", help="Scenario CSV with Scenario, Weight, GrossCAGR, optional StressWeight.")
    parser.add_argument("--portfolio", help="Portfolio CSV with Ticker, Weight, ExpenseRatio.")
    parser.add_argument("--portfolio-er", type=str, help="Direct portfolio expense ratio, e.g. 0.0042 or 0.42%.")
    parser.add_argument("--output", help="Write JSON output to this file. If omitted, prints to stdout.")

    parser.add_argument("--scenario-col", default="Scenario")
    parser.add_argument("--weight-col", default="Weight")
    parser.add_argument("--gross-cagr-col", default="GrossCAGR")
    parser.add_argument("--stress-weight-col", default="StressWeight")

    parser.add_argument("--portfolio-weight-col", default="Weight")
    parser.add_argument("--portfolio-er-col", default="ExpenseRatio")

    parser.add_argument("--horizon-years", type=int, default=40)
    parser.add_argument("--annual-draw-rate", default="0.03")
    parser.add_argument("--initial-wealth", default="1.0")
    parser.add_argument("--draw-timing", choices=["year_end", "year_begin"], default="year_end")
    parser.add_argument("--allow-missing-stress-blend", action="store_true",
                        help="If StressWeight is missing, re-normalize blend over available canonical/equal legs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.portfolio and args.portfolio_er is None:
        parser.error("Provide either --portfolio or --portfolio-er.")

    scenario_rows = read_csv_dicts(args.scenarios)

    if args.portfolio_er is not None:
        er = parse_rate(args.portfolio_er)
    else:
        portfolio_rows = read_csv_dicts(args.portfolio)
        er = weighted_expense_ratio(
            portfolio_rows,
            weight_col=args.portfolio_weight_col,
            expense_ratio_col=args.portfolio_er_col,
        )

    cfg = RCEW40Config(
        horizon_years=args.horizon_years,
        annual_draw_rate=parse_rate(args.annual_draw_rate),
        initial_wealth=float(str(args.initial_wealth).replace(",", "")),
        draw_timing=args.draw_timing,
        require_stress_for_robust_blend=not args.allow_missing_stress_blend,
    )

    result = compute_rcew40(
        scenario_rows,
        portfolio_expense_ratio=er,
        scenario_col=args.scenario_col,
        canonical_weight_col=args.weight_col,
        gross_cagr_col=args.gross_cagr_col,
        stress_weight_col=args.stress_weight_col,
        config=cfg,
    )

    output_obj = asdict(result)

    if args.output:
        write_json(args.output, output_obj)
    else:
        print(json.dumps(output_obj, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
