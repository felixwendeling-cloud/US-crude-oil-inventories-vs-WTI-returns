"""
US crude oil inventories vs their seasonal norm: does the gap predict WTI returns?

Data (free, no API key, both from the US Energy Information Administration):
  - Weekly US commercial crude stocks excluding SPR   (EIA series WCESTUS1)
  - WTI crude oil spot price, daily                   (EIA series RWTC; FRED DCOILWTICO as backup)

Method in one paragraph:
  For each week, compare inventories with the average for the same week over the
  PREVIOUS five years only. Treat the signal as public only on the Wednesday after
  the Friday the week ends (EIA release), enter at the first price on or after that
  date, and measure forward returns. Test with a regression using Newey-West (HAC)
  standard errors, check it out of sample, then run a simple long/short rule.

Run:  python crude_inventory_signal.py
If a download is blocked, get the files by hand (see README) and save them next to
this script as stocks.xls (or .xlsx/.csv) and prices.xls (or .xlsx/.csv).
In Colab, "next to the script" means the /content folder (file panel on the left).
"""
from pathlib import Path
import io
import sys

import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

UA = {"User-Agent": "Mozilla/5.0 (research project; python-requests)"}
STOCK_URLS = ["https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"]
PRICE_URLS = ["https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls",
              "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"]
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()   # works in Colab and as a script
SPLIT = "2019-01-01"          # train before, test from this date
HORIZONS = (1, 4)             # forward return horizons in weeks
Z_THRESHOLD = 1.0             # signal threshold for the long/short rule


def _fetch(urls, label):
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return io.BytesIO(r.content)
            print(f"{label}: {url} returned HTTP {r.status_code}")
        except Exception as exc:
            print(f"{label}: could not reach {url} ({exc})")
    return None


def _parse(src):
    """Read an EIA Excel file or a FRED-style CSV into a clean date-indexed Series."""
    readers = [
        lambda: pd.read_excel(src, sheet_name="Data 1", header=None),   # EIA layout
        lambda: pd.read_excel(src, sheet_name=0, header=None),
        lambda: pd.read_csv(src, header=None),                          # FRED layout
    ]
    raw = None
    for read in readers:
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            raw = read()
            break
        except Exception:
            continue
    if raw is None:
        raise ValueError("unrecognised file format")
    first = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    header_rows = first[first.isin(["date", "observation_date"])].index
    if len(header_rows) == 0:
        raise ValueError("could not find a 'Date' header row")
    data = raw.iloc[header_rows[0] + 1:, :2].copy()
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    return data.dropna().drop_duplicates("date").set_index("date")["value"].sort_index()


def load(label, local_stem, urls):
    for ext in (".xls", ".xlsx", ".csv"):
        local = HERE / f"{local_stem}{ext}"
        if local.exists():
            print(f"{label}: using local file {local.name}")
            return _parse(str(local))
    src = _fetch(urls, label)
    if src is None:
        sys.exit(
            f"\nCould not download {label}. Download it by hand (see README) and save it as "
            f"{local_stem}.xls, {local_stem}.xlsx or {local_stem}.csv in {HERE}, then run again."
        )
    return _parse(src)


def build_features(stocks, prices):
    df = stocks.to_frame("stocks")
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.values

    # Seasonal norm: mean of the same ISO week over the previous 5 years (current year excluded)
    piv = df.pivot_table(index="year", columns="week", values="stocks")
    avg5 = piv.shift(1).rolling(5).mean()
    df["avg5"] = [avg5.loc[y, w] for y, w in zip(df["year"], df["week"])]
    df["dev"] = (df["stocks"] - df["avg5"]) / df["avg5"]

    # The Friday-dated figure is only public on the Wednesday after (5 days later)
    df["available"] = df.index + pd.Timedelta(days=5)
    entry = prices.reindex(df["available"], method="bfill").values

    for h in HORIZONS:
        exit_ = prices.reindex(df["available"] + pd.Timedelta(weeks=h), method="bfill").values
        ret = exit_ / entry - 1
        ret[(entry <= 0) | (exit_ <= 0)] = np.nan      # WTI traded below zero in April 2020
        df[f"fwd{h}w"] = ret

    # z-score of the deviation using only information available up to each date
    mean = df["dev"].expanding(min_periods=104).mean()
    std = df["dev"].expanding(min_periods=104).std()
    df["z"] = (df["dev"] - mean) / std
    return df.dropna(subset=["dev", "z"])


def regression_report(df, h):
    data = df.dropna(subset=[f"fwd{h}w"])
    train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
    res = smf.ols(f"fwd{h}w ~ dev", data=train).fit(cov_type="HAC", cov_kwds={"maxlags": h + 2})
    pred = res.predict(test)
    realised = test[f"fwd{h}w"]
    nonzero = realised != 0
    return {
        "horizon_weeks": h,
        "n_train": len(train),
        "n_test": len(test),
        "slope": res.params["dev"],
        "t_stat_HAC": res.tvalues["dev"],
        "p_value": res.pvalues["dev"],
        "r2_train": res.rsquared,
        "oos_corr": float(np.corrcoef(pred, realised)[0, 1]),
        "oos_hit_rate": float((np.sign(pred[nonzero]) == np.sign(realised[nonzero])).mean()),
    }


def strategy_report(df):
    """Weekly rule: long when stocks are unusually LOW, short when unusually HIGH. Gross of costs."""
    data = df.dropna(subset=["fwd1w"])
    test = data[data.index >= SPLIT].copy()
    test["pos"] = np.where(test["z"] < -Z_THRESHOLD, 1, np.where(test["z"] > Z_THRESHOLD, -1, 0))
    test["strat"] = test["pos"] * test["fwd1w"]

    def stats(r):
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan
        return {"ann_return": r.mean() * 52, "sharpe": sharpe, "max_drawdown": dd}

    out = {
        "n_weeks": len(test),
        "weeks_in_market": int((test["pos"] != 0).sum()),
        "strategy": stats(test["strat"]),
        "buy_and_hold": stats(test["fwd1w"]),
    }
    return out, test


def make_charts(df, test):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(df.index, df["dev"] * 100, lw=0.8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvline(pd.Timestamp(SPLIT), color="r", ls="--", lw=0.8, label="train | test")
    ax[0].set_title("US commercial crude stocks vs 5-year seasonal average (%)")
    ax[0].legend()
    ax[1].plot(test.index, (1 + test["strat"]).cumprod(), label="Rule (gross)")
    ax[1].plot(test.index, (1 + test["fwd1w"]).cumprod(), label="Buy and hold WTI", alpha=0.7)
    ax[1].set_title("Out-of-sample: weekly long/short rule vs buy and hold")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(HERE / "charts.png", dpi=150)


def write_results(regs, strat):
    lines = ["# Results (generated by crude_inventory_signal.py)", "",
             f"Train: before {SPLIT}. Test: from {SPLIT}. Returns gross of costs.", "",
             "## Regression: forward return on inventory deviation (HAC standard errors)", "",
             "| Horizon (wks) | Slope | t-stat | p-value | OOS correlation | OOS hit rate | n train | n test |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in regs:
        lines.append(f"| {r['horizon_weeks']} | {r['slope']:.3f} | {r['t_stat_HAC']:.2f} | {r['p_value']:.3f} | "
                     f"{r['oos_corr']:.3f} | {r['oos_hit_rate']:.1%} | {r['n_train']} | {r['n_test']} |")
    s, b = strat["strategy"], strat["buy_and_hold"]
    lines += ["", f"## Weekly long/short rule (|z| > {Z_THRESHOLD}), out of sample", "",
              f"Weeks in test: {strat['n_weeks']}, weeks with a position: {strat['weeks_in_market']}", "",
              "| | Annualised return | Sharpe | Max drawdown |", "| --- | --- | --- | --- |",
              f"| Rule | {s['ann_return']:.1%} | {s['sharpe']:.2f} | {s['max_drawdown']:.1%} |",
              f"| Buy and hold | {b['ann_return']:.1%} | {b['sharpe']:.2f} | {b['max_drawdown']:.1%} |"]
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    stocks = load("EIA weekly crude stocks", "stocks", STOCK_URLS)
    prices = load("WTI spot price", "prices", PRICE_URLS)
    df = build_features(stocks, prices)
    regs = [regression_report(df, h) for h in HORIZONS]
    strat, test = strategy_report(df)
    make_charts(df, test)
    write_results(regs, strat)
    print((HERE / "results.md").read_text(encoding="utf-8"))
    print("\nSaved charts.png and results.md")


if __name__ == "__main__":
    main()
