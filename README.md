# The Paper Phase Transition

**Vasion challenge · Intermountain Quantum Competition 2026 · BYU, November 6–7**

Industries don't go digital gradually. They hold out, then flip once enough of their partners flip. A hospital stops faxing when its insurers stop accepting faxes. A law firm drops wet signatures when the banks and courts it works with go digital.

Physicists describe this kind of all-at-once change with the **Ising model**, and quantum computers are well suited to Ising problems. Quantum annealers such as D-Wave's solve them natively.

## The challenge

**Use quantum computing to predict when each industry tips from paper to digital, and which industries pull others along.**

---

## The model in 60 seconds

| Piece | What it means | Symbol |
|---|---|---|
| **Node** | One industry in one region, such as *Healthcare in Texas* | *i* |
| **State** | Paper or digital | *sᵢ* = −1 (paper) or +1 (digital) |
| **Tie** | Two nodes that do business together. A strong tie means "if you flip, I want to flip too." | *Jᵢⱼ* ≥ 0 |
| **Resistance** | How hard a node holds on to paper: regulation, wet-signature rules, and habit | *rᵢ* |
| **AI pressure** | Document AI keeps getting cheaper and better, so the push toward digital grows every year | *A*(year) |

Each year, the economy settles into the state with the **lowest energy**:

```
Energy(year) = − Σ  Jᵢⱼ · sᵢ · sⱼ   −   Σ  (A(year) − rᵢ) · sᵢ
                 ties                    nodes
```

- **First term:** partners lower the energy by matching, so ties pull neighbors toward the same state.
- **Second term:** once AI pressure is greater than a node's resistance, being digital lowers the energy.

Solve it once per year, say 2026 to 2040. The first year a node comes out digital is its **flip year**. Put every flip year on a map and you have a **tipping-point map**.

**A quick example.** Suppose Healthcare in Texas has resistance 0.75. On its own, it would flip once AI pressure passes 0.75. But Healthcare is tied tightly to Insurance, which flipped years earlier, and that tie pulls Healthcare over the line sooner. Ties can also hold a node back: an eager industry surrounded by stubborn partners flips later than it would alone. Those stories are the heart of a great presentation.

---

## What to present

Judging is based entirely on your final presentation (about 5 minutes). Cover these four things:

1. **Your model.** What your nodes, ties, and resistances are, and where your numbers came from.
2. **Your tipping-point map from a quantum approach.** When does each industry flip? Is there a big tipping year? Any platform is fine, on real hardware or a simulator.
3. **A classical comparison.** Run the same model with a classical method and compare: do they agree, how long does each take, and what happens as the network grows?
4. **The story.** Who tips first, who pulls whom, and what a business should do with this. Also tell us where the quantum hardware helped and where it didn't.

**Stretch goal: where should Vasion sell?** Once you have flip years, choose which industries and regions a sales team should target *before* they flip, within a limited sales budget. Landing two tied industries is worth more than landing either alone. That's a second QUBO for your quantum solver.

> **Be honest about the classical side.** For a network this size, a laptop can solve this model exactly in under a second. Nobody expects you to beat it. Winning teams show their quantum approach reproduces the tipping points, test how it scales, and explain clearly what they learned about turning a real business question into an Ising model.

### How judges score

| Weight | What we look for |
|---|---|
| 25% | **Model:** sensible nodes, ties, and resistances, with sources or reasoning |
| 25% | **Quantum results:** a working tipping-point map from your quantum approach |
| 20% | **Classical comparison and scaling:** an honest side-by-side |
| 30% | **Story and insight:** clear explanations of who tips, who pulls whom, and what it means |

---

## A starting network (use it, change it, or replace it)

You build your own network. If you want a head start, this one gives an interesting map: flips spread over about ten years, with one big tipping year.

**20 industries and a suggested base resistance** (0 = eager, 1 = stubborn):

| Industry | Resistance | Industry | Resistance |
|---|---|---|---|
| Technology | 0.10 | Insurance | 0.55 |
| Media | 0.20 | Energy | 0.55 |
| Telecom | 0.25 | Pharma | 0.60 |
| Retail | 0.30 | Education | 0.60 |
| Hospitality | 0.35 | Nonprofit | 0.65 |
| Logistics | 0.40 | Real Estate | 0.65 |
| Banking | 0.45 | Construction | 0.70 |
| Accounting | 0.50 | Agriculture | 0.70 |
| Manufacturing | 0.50 | Healthcare | 0.75 |
| | | Government | 0.85 |
| | | Legal | 0.90 |

**Partner ties within a region** (strength 0–1): Healthcare–Insurance 1.0 · Manufacturing–Logistics 0.9 · Banking–Real Estate 0.8 · Retail–Logistics 0.8 · Construction–Real Estate 0.8 · Healthcare–Pharma 0.7 · Legal–Real Estate 0.7 · Healthcare–Government 0.6 · Legal–Government 0.6 · Accounting–Banking 0.6 · Government–Education 0.6 · Agriculture–Logistics 0.6 · Telecom–Technology 0.6 · Insurance–Banking 0.5 · Legal–Banking 0.5 · Energy–Manufacturing 0.5 · Technology–Media 0.5 · Insurance–Legal 0.4 · Accounting–Nonprofit 0.4 · Construction–Government 0.4 · Hospitality–Retail 0.4 · Pharma–Logistics 0.4 · Nonprofit–Education 0.4 · Media–Retail 0.3

**Regions:** about a dozen U.S. regions (Pacific NW, California, Mountain West, Southwest, Texas, and so on). The same industry in neighboring regions shares vendors and regulators, so connect them with a weak tie.

**Values that worked well in testing:**

| Setting | Suggested value |
|---|---|
| AI pressure | Starts at 0 in 2026 and rises 0.07 per year |
| Partner tie *Jᵢⱼ* | 0.15 × the strength above |
| Same industry, neighboring region | 0.05 |
| Resistance per node | Industry value, plus a regional shift (about ±0.15), plus a little random noise |

Stronger ties make whole clusters flip in the same year. A strong cross-region tie makes every region flip in lockstep. Both are worth playing with, and both are worth explaining.

Better numbers make a better story. Public data on paper use, e-signature laws, or digital adoption by industry is fair game. Tell us what you used.

---

## Tech suggestions

These are suggestions, not requirements. Use whatever tools and platforms you like, and arrange your own accounts for any cloud service.

| Need | Suggestion |
|---|---|
| A quantum annealer (a natural fit) | D-Wave's documentation: [Ocean SDK](https://docs.dwavequantum.com/en/latest/ocean/index.html) and [Leap cloud service](https://docs.dwavequantum.com/en/latest/leap_sapi/) |
| A gate-based approach | Write the same model as a QUBO and solve it with QAOA, using the SDK of any sponsor platform at the event |
| Classical comparison | Simulated annealing runs on a laptop with the same model |
| Exact answer to check against | Because every tie is positive, a minimum cut gives the exact lowest-energy state. `networkx` has `minimum_cut`. |
| Networks and charts | `networkx`, `pandas`, `matplotlib` |

**Watch the signs.** Most solvers, including Ocean, minimize `Σ hᵢ·sᵢ + Σ Jᵢⱼ·sᵢ·sⱼ`, which is the opposite sign from the formula above. To make partners want to match and pressure push toward digital, use **h = −(A − r)** and **J = −(tie strength)**. If everything flips in 2026, or nothing ever flips, check this first.

**More tips**

- Get the model working on one region (20 nodes) before you scale to every region.
- Your network doesn't change from year to year; only the pressure does. On an annealer, find the embedding once and reuse it.
- The ties are small compared to the pressure terms, and quantum hardware has limited precision, so a few nodes may come out wrong. Measuring that is a good result. Working around it is a better one.
- A heatmap of industries (rows) by regions (columns), colored by flip year, makes a great slide.
- Use simulated annealing or a simulator to iterate quickly, and save real hardware time for final runs.

---

## Rules

1. Use a quantum approach for at least part of your work (real hardware or a simulator, on any platform).
2. Include a classical comparison.

## Mentors

Justin Fife and Adam Cartwright from Vasion are on site both days. Come find us with questions about the model, quantum tools, or your story.
