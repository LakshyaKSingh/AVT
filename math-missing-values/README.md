# Detecting & Imputing Missing Values

Beginner | data-processing

### The problem, from first principles

**Detecting missing values**

Real datasets are never as clean as the small, hand-picked arrays this curriculum's Math & Statistics tracks used, a survey respondent skips a question, a sensor drops a reading, a merge between two tables leaves some rows without a match. Before anything else, `03-mse-gradient`, PCA, a Gaussian's MLE, all assume every entry of the data actually holds a real number, a single missing value silently poisons a mean, a covariance, a gradient, everywhere it touches.

The very first, unglamorous step of any real ML pipeline is knowing exactly where the gaps are, before deciding what to do about them (`02-imputing-missing-values`, the next question, is that "what to do about them" step). This question is purely about detection: finding missing entries reliably, a surprisingly easy thing to get subtly wrong.

**Imputing missing values**

`01-detecting-missing-values` found the gaps. Now you need to fill them, most ML operations (a matrix multiply, a gradient computation, a distance calculation) simply cannot proceed with a `NaN` sitting in the middle of an array, it poisons every computation that touches it. The simplest reasonable fill-in value for a missing number is "whatever's typical for that column," and there are two natural choices for "typical": the mean, and the median.

The choice between them isn't arbitrary, it's the exact same distinction `02-summarizing-a-distribution` (the next track) makes between mean and median as measures of central tendency: one is sensitive to outliers, one isn't, and that sensitivity carries straight through into how good your imputed values end up being.

### From theory to code

**Detecting missing values**

Theory names the standard representation for a missing numeric value (`np.nan`) and flags the single most common bug in detecting it (`== np.nan` never works, by design). Implement a boolean mask using the correct tool, then column-wise counts and fractions built directly from that mask.

Implement `missing_mask(x)`, `missing_count_per_column(x)` and `missing_fraction_per_column(x)` against that reasoning. The signatures and docstrings are already in the editor.

**Imputing missing values**

Theory fills each missing value with its own COLUMN's mean or median, computed only from that column's actually-observed values (ignoring the missing ones, not accidentally treating them as zero). Implement both, reusing `01-detecting-missing-values`'s mask to find exactly where to write the fill-in values.

Implement `impute_with_mean(x)` and `impute_with_median(x)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

**Detecting missing values**

- `x` is a 2D float array, `(num_rows, num_columns)`, missing values represented as `np.nan`.
- Use `np.isnan`, never `== np.nan` (Theory explains exactly why the latter silently fails).
- `missing_fraction_per_column` returns fractions in `[0.0, 1.0]`, one per column.

**Imputing missing values**

- Neither function may mutate the input array `x`, work on a copy.
- Compute each column's mean/median from ONLY that column's non-missing values (`np.nanmean`/`np.nanmedian` do this automatically).
- A value's replacement must come from its OWN column, not some other column's statistic.

### Hints

**Detecting missing values**

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.isnan(x)` already returns a same-shaped boolean mask directly, no comparison operator needed.

</details>

<details>
<summary>Hint 2</summary>

Once you have the mask, `missing_count_per_column` is `.sum(axis=0)` (summing down each column, over all rows).

</details>

**Imputing missing values**

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.nanmean(x, axis=0)` computes a length-`num_columns` array of column means, already ignoring NaNs, in one call.

</details>

<details>
<summary>Hint 2</summary>

`missing_mask(x)` (from `01-detecting-missing-values`) tells you exactly which positions to overwrite; `np.where(mask)` gives you the row and column index of each one.

</details>
