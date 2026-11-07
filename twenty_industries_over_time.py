"""Find flip years for the README's 20 industries in one example region.

This classical solver checks every arrangement using NumPy. It is intended
for this small learning example; enumeration grows exponentially with nodes.
The inputs are the README's suggestions, with no regional shift or noise.

Run with: python twenty_industries_over_time.py
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np


# Each industry now has its own resistance, rather than a separate variable.
resistances = {
    "Technology": 0.10,
    "Media": 0.20,
    "Telecom": 0.25,
    "Retail": 0.30,
    "Hospitality": 0.35,
    "Logistics": 0.40,
    "Banking": 0.45,
    "Accounting": 0.50,
    "Manufacturing": 0.50,
    "Insurance": 0.55,
    "Energy": 0.55,
    "Pharma": 0.60,
    "Education": 0.60,
    "Nonprofit": 0.65,
    "Real Estate": 0.65,
    "Construction": 0.70,
    "Agriculture": 0.70,
    "Healthcare": 0.75,
    "Government": 0.85,
    "Legal": 0.90,
}

# These are the raw connection strengths from the README.
partner_strengths = [
    ("Healthcare", "Insurance", 1.0),
    ("Manufacturing", "Logistics", 0.9),
    ("Banking", "Real Estate", 0.8),
    ("Retail", "Logistics", 0.8),
    ("Construction", "Real Estate", 0.8),
    ("Healthcare", "Pharma", 0.7),
    ("Legal", "Real Estate", 0.7),
    ("Healthcare", "Government", 0.6),
    ("Legal", "Government", 0.6),
    ("Accounting", "Banking", 0.6),
    ("Government", "Education", 0.6),
    ("Agriculture", "Logistics", 0.6),
    ("Telecom", "Technology", 0.6),
    ("Insurance", "Banking", 0.5),
    ("Legal", "Banking", 0.5),
    ("Energy", "Manufacturing", 0.5),
    ("Technology", "Media", 0.5),
    ("Insurance", "Legal", 0.4),
    ("Accounting", "Nonprofit", 0.4),
    ("Construction", "Government", 0.4),
    ("Hospitality", "Retail", 0.4),
    ("Pharma", "Logistics", 0.4),
    ("Nonprofit", "Education", 0.4),
    ("Media", "Retail", 0.3),
]

tie_scale = 0.15
ties = [(first, second, tie_scale * strength)
        for first, second, strength in partner_strengths]
start_year = 2026
end_year = 2040
pressure_per_year = 0.07


def prepare_arrangements(resistances, ties):
    """List every arrangement and calculate the part of energy fixed over time."""
    industries = list(resistances)
    indices = {industry: index for index, industry in enumerate(industries)}
    arrangement_count = 2 ** len(industries)
    numbers = np.arange(arrangement_count, dtype=np.uint32)
    states = np.empty((arrangement_count, len(industries)), dtype=np.int8)

    # Each bit represents one industry's choice: 0 -> paper, 1 -> digital.
    for index in range(len(industries)):
        states[:, index] = 2 * ((numbers >> index) & 1).astype(np.int8) - 1

    # E = -sum(J*s_i*s_j) + sum(r_i*s_i) - pressure*sum(s_i).
    # Only the final term changes each year, so reuse the other terms.
    fixed_energy = states @ np.array(list(resistances.values()))
    for first, second, strength in ties:
        fixed_energy -= strength * states[:, indices[first]] * states[:, indices[second]]
    spin_totals = states.sum(axis=1, dtype=np.int32)
    return industries, states, fixed_energy, spin_totals


def solve_year(pressure, states, fixed_energy, spin_totals):
    """Return each industry's state, preserving ambiguity in equal-energy answers."""
    energies = fixed_energy - pressure * spin_totals
    lowest_energy = float(energies.min())
    winners = states[np.isclose(energies, lowest_energy, rtol=0.0, atol=1e-9)]
    labels = []
    for index in range(states.shape[1]):
        if np.all(winners[:, index] == 1):
            labels.append("Digital")
        elif np.all(winners[:, index] == -1):
            labels.append("Paper")
        else:
            labels.append("Either")
    return labels, lowest_energy


def main():
    started = perf_counter()
    industries, states, fixed_energy, spin_totals = prepare_arrangements(resistances, ties)
    flip_years = {industry: None for industry in industries}
    rows = []
    found_tie = False

    print(f"One example region: {len(industries)} industries, {len(ties)} partner ties.")
    print(f"Method: classical exhaustive search over {len(states):,} arrangements.")
    print("Inputs: suggested README values, without regional shifts or noise.\n")
    print(f"{'Year':<6} {'Pressure':>8} {'Digital':>8}  Newly digital")

    for year in range(start_year, end_year + 1):
        pressure = pressure_per_year * (year - start_year)
        labels, lowest_energy = solve_year(pressure, states, fixed_energy, spin_totals)
        newly_digital = []
        for industry, label in zip(industries, labels):
            if label == "Digital" and flip_years[industry] is None:
                flip_years[industry] = year
                newly_digital.append(industry)
            found_tie = found_tie or label == "Either"
            rows.append({"year": year, "pressure": round(pressure, 2),
                         "industry": industry, "state": label,
                         "lowest_energy": round(lowest_energy, 10)})

        print(f"{year:<6} {pressure:>8.2f} {labels.count('Digital'):>8}  "
              f"{', '.join(newly_digital) or '-'}")

    elapsed = perf_counter() - started
    print("\nFirst definite digital year:")
    for industry, year in sorted(flip_years.items(), key=lambda item: (item[1] or 9999, item[0])):
        print(f"{industry:<16} {year if year is not None else 'Not reached'}")
    if found_tie:
        print("Either means equally good arrangements disagree on that industry.")
        print("Flip years require all lowest-energy arrangements to agree on digital.")
    print(f"\nPreparation and yearly solving: {elapsed:.3f} seconds (includes console output).")

    output_path = Path(__file__).with_name("twenty_industries_results.csv")
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Yearly states saved to: {output_path}")


if __name__ == "__main__":
    main()
