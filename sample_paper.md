# Depth-Aware Evaluation of Long-Context Language Models

## Abstract

Existing long-context benchmarks evaluate models with isotropic difficulty across context positions, treating retrieval at depth 5K identically to retrieval at depth 500K. This isotropic framing obscures a clinically important pattern: most production failures occur not at maximum depth, but at intermediate depths where models exhibit unexpected non-monotonic accuracy curves. We introduce depth-stratified evaluation, sampling 64 needle positions across 12 contiguous depth strata up to 200K tokens, and apply it to six frontier models. We find that four of six models show non-monotonic accuracy-vs-depth curves, with the worst case (Model X) reaching peak accuracy at 32K depth before declining to 41% at 180K depth. Synthetic-only benchmarks systematically over-estimate models' usable context windows by a median of 38%. We release our benchmark and probe code openly, and discuss implications for prompt engineering and context budgeting in production.

## 1. Introduction

Long-context language models advertise context windows from 200K to 2M tokens, with implicit suggestions of uniform retrieval performance across that range. In production, engineers report a different reality: model behavior degrades unevenly across depths, with surprising local maxima and minima. The discrepancy between advertised and observed context performance has measurable cost — wasted tokens in over-provisioned prompts, retrieval misses when content is placed at "bad" depths, and hard-to-diagnose failures when retrieval works in development at one depth but fails in production at another.

The dominant evaluation paradigm, exemplified by the needle-in-a-haystack (NIAH) family of benchmarks, samples needle positions uniformly across the context. This isotropic sampling assumes that retrieval difficulty is a smooth function of depth. Recent observations suggest otherwise: many models exhibit non-monotonic accuracy curves, with intermediate depths producing failures that maximum depths do not. The existing literature has not systematically characterized these patterns.

In this work, we propose depth-stratified evaluation. Rather than sampling needle positions uniformly, we partition the context into 12 contiguous strata and sample 64 positions within each stratum. The resulting accuracy-vs-depth curve exposes non-monotonicity that uniform sampling masks. We apply this methodology to six frontier models and report systematic patterns of non-monotonic decay. We then quantify the gap between advertised and effective context windows, showing that the latter is systematically smaller and that the size of the gap varies substantially across models with comparable headline scores.

## 2. Method

We use a controlled paraphrase-needle setup: each evaluation example consists of a multi-paragraph haystack drawn from an open corpus, with a single fact-bearing needle paraphrased and inserted at a chosen depth. The model is asked a direct retrieval question whose answer is the inserted needle. We measure exact-match accuracy and report per-stratum results.

Context lengths probed: 16K, 32K, 64K, 128K, 200K tokens. Depth strata: 12 contiguous regions per context length. Needle count per stratum: 64. Total examples per model per context length: 768. Total examples across the study: approximately 13,800.

Models evaluated: GPT-4 family (3 variants), Claude family (2 variants), Gemini (1 variant). Where applicable, prompt caching was used to amortize haystack reuse across needle positions.

To support reproducibility, we publish the probe code and the haystack corpus under an MIT license, with adapter code for six provider APIs.

## 3. Results

We summarize three findings.

**Finding 1: Non-monotonic decay is the rule, not the exception.** Four of six models exhibited at least one non-monotonic transition across depths in their 200K runs. Two models showed local maxima at depths between 16K and 64K, declining at both shallower and deeper depths. We refer to this pattern as "interior peaks." Interior peaks are particularly costly in practice because they violate the intuition that "placing the needle closer to the start helps" — a heuristic widely used in prompt engineering.

**Finding 2: Effective context is overestimated by a median of 38% in synthetic benchmarks.** Defining "effective context" as the longest depth at which a model retains 90% of its peak accuracy, we found systematic discrepancies between the advertised window and the depth at which performance collapsed. Model X advertises a 1M token window; its effective window under our methodology is 192K. Model Y advertises 200K; its effective window is 124K. The over-estimate ratio varied from 1.18× (best case) to 5.21× (worst case), with a median of 1.62×.

**Finding 3: Models with similar peak accuracy diverge sharply in mid-depth behavior.** Two models with identical 95% peak retrieval at 32K showed dramatically different curves at 128K: one held at 87%, the other collapsed to 52%. This divergence is invisible in benchmarks that average across positions. For applications where retrieval reliability matters more than peak performance, this suggests that current leaderboard scores are misleading.

For practitioners, depth-stratified evaluation is a more honest predictor of production behavior than uniform NIAH. For prompt engineering, the existence of interior peaks suggests that content placement matters in ways prior work has under-emphasized. For model selection, two models with equal "long-context performance" scores may differ by a factor of 1.5× in usable context.

## 4. Limitations

First, our needle setup uses paraphrased facts, which is one specific retrieval task; other long-context tasks (reasoning, aggregation, code completion) may exhibit different depth profiles. Second, we evaluate at six discrete context lengths; finer-grained sampling would produce smoother curves. Third, our haystack corpus is in English; behavior may differ in other languages. Fourth, we report exact-match accuracy; partial-credit or semantic-similarity metrics might tell a different story. Fifth, prompt caching adoption varies across providers; we did our best to apply consistent caching settings, but absolute latency comparisons should be read with this caveat. Sixth, we did not control for model release date; some of the variation across models may reflect ongoing improvements rather than fundamental architectural differences.

## 5. Conclusion

Long-context models are not uniformly accurate across their advertised context windows. Depth-stratified evaluation makes the non-monotonicity visible and produces more honest measures of effective context. We release our probe code openly and encourage adoption as a complement to uniform-sampling benchmarks. Future work should extend depth-stratified evaluation beyond retrieval to long-context reasoning and aggregation tasks, where the depth dependence may be even sharper.
