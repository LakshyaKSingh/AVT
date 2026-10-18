# THE ROLLOUT CHECK

Beginner | probability-and-statistics

**Difficulty:** Easy
**Tags:** Probability & Statistics

---

### Story

Robinhood A/B-tests a new order-ticket UI against the existing one. Before anyone calls the
difference "real," it needs a proper two-proportion z-test, not just eyeballing two percentages.

---

### The Math

```
p_hat = (x1 + x2) / (n1 + n2)
SE = sqrt( p_hat * (1 - p_hat) * (1/n1 + 1/n2) )
z = (p2 - p1) / SE
```

Two-sided p-value from the standard normal CDF, where `p1 = x1/n1` and `p2 = x2/n2` (group 1 =
control, group 2 = treatment).

### Input Format

```
n1 x1 n2 x2
```

### Output Format

`p1 p2 z pvalue`, 6 decimals.

### Constraints

- `1 <= n1, n2 <= 10^7`, `0 <= x_i <= n_i`
- Time limit: 1.0 second.

---

### Example

**Input**

```
5000 250 5000 300
```

**Output**

```
0.050000 0.060000 2.193172 0.028295
```
