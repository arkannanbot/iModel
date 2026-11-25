"""Generate the sample golden dataset: normalise free-form English dates to ISO 8601.

Small models get the format wrong out of the box, but it is learnable in a few minutes
of LoRA training on CPU, which makes it a good end-to-end smoke test.

    python scripts/make_sample_data.py   # writes data/sample/{train,eval,test}.jsonl
"""
import calendar
import json
import random
from pathlib import Path

MONTHS = list(calendar.month_name)[1:]
ABBR = list(calendar.month_abbr)[1:]


def ordinal(n: int) -> str:
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def render(rng: random.Random, y: int, m: int, d: int) -> str:
    fmts = [
        lambda: f"{MONTHS[m-1]} {d}, {y}",
        lambda: f"{d} {MONTHS[m-1]} {y}",
        lambda: f"{ABBR[m-1]} {ordinal(d)}, {y}",
        lambda: f"the {ordinal(d)} of {MONTHS[m-1]}, {y}",
        lambda: f"{m}/{d}/{y}",
        lambda: f"{y}.{m:02d}.{d:02d}",
        lambda: f"{MONTHS[m-1].upper()} {d} {y}",
        lambda: f"{calendar.day_name[calendar.weekday(y, m, d)]}, {MONTHS[m-1]} {ordinal(d)} {y}",
    ]
    date = rng.choice(fmts)()
    wraps = ["{}", "Date: {}", "The meeting is on {}.", "Invoice issued {}", "{} (please confirm)"]
    return rng.choice(wraps).format(date)


def main() -> None:
    rng = random.Random(7)
    seen, rows = set(), []
    while len(rows) < 520:
        y, m = rng.randint(1950, 2035), rng.randint(1, 12)
        d = rng.randint(1, calendar.monthrange(y, m)[1])
        text = render(rng, y, m, d)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"input": text, "output": f"{y:04d}-{m:02d}-{d:02d}"})
    out = Path(__file__).resolve().parent.parent / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", rows[:400]), ("eval", rows[400:460]), ("test", rows[460:])):
        with open(out / f"{name}.jsonl", "w") as f:
            for i, r in enumerate(chunk):
                f.write(json.dumps({"id": f"{name}-{i}", **r}) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
