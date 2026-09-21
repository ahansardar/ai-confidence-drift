# AI Confidence Drift Detection Using Machine Learning

**Organization:** PranavX Labs
**Internship Domain:** Artificial Intelligence & Machine Learning
**Project Type:** AI/ML Research + Experimental Implementation

---

## 1. Abstract

AI models usually give a confidence score along with their prediction, but that confidence doesn't always match how correct the model actually is. In this project I wanted to check if changes in a model's confidence, tracked across a series of related predictions, can tell us something useful about whether a prediction should be trusted.

I trained a simple text classifier (TF-IDF + Logistic Regression) on a public two class newsgroup dataset. To actually get a "series" of confidence values for each sample, I fed every test document to the model multiple times, each time with a bit more of the text corrupted. This gave me real confidence sequences, not made up numbers. From these sequences I built statistical features (moving average, volatility, how many times confidence dropped in a row, and so on) and used them to train a second model whose job is to guess if a prediction is unreliable.

The results were mixed. On the 439 documents, net confidence change had a correlation of 0.090 with a correct prediction becoming incorrect. In five-fold document-grouped evaluation, adding confidence at the current step raised unreliable-prediction F1 from 0.296 to 0.468. Adding the nine history features raised it further to 0.508; ROC-AUC rose from 0.729 to 0.763 over the current-confidence model. The gain from history was modest but positive in this experiment. The base model's expected calibration error was 0.153 and its binary Brier score was 0.160. Six of 100 high-confidence predictions were wrong.

The original per-step drift detector did not flag any of those six errors at the uncorrupted step, where no history exists. I then trained a sequence-level detector to predict whether the *original* classification was wrong after six controlled probes. It flagged all six on this dataset, along with 50 correct high-confidence predictions. A separate immediate review rule also covered all six, with 78 correct cases sent to review. The new detector improves coverage and review cost here, but the result is retrospective and needs confirmation on a new corpus.

## 2. Introduction

A lot of ML systems in the real world are trusted because they report a high confidence number. If that confidence isn't actually reliable, that trust can lead to bad decisions downstream, for example auto-approving something that should have gone to a human for review. In this project I treat the model's confidence as something to test, not something to assume is correct.

## 3. Problem Statement

A model's stated confidence is not guaranteed to reflect how likely it is to actually be right. It can stay high even when the prediction is wrong, or it can drop for reasons that have nothing to do with correctness. Before this project there wasn't a clear method here for using changes in confidence across repeated predictions to flag predictions that are likely unreliable.

## 4. Research Question

Can machine learning detect meaningful changes in an AI model's confidence, and can those changes be used to identify predictions that are likely unreliable?

I also wanted to answer a few smaller questions along the way:
- How stable is confidence when the input is only slightly changed?
- Does confidence tend to drop before a prediction becomes wrong?
- Can confidence drift actually be used to predict errors?
- Are high confidence but wrong predictions something I can catch?
- Which features actually help detect drift, and which ones don't?

## 5. Hypothesis

My hypothesis (H1) was that statistical features built from a confidence sequence, things like moving average, volatility, and consecutive drops, would predict unreliable predictions better than the raw confidence value alone, and much better than not using confidence at all.

The experiments below separate the gain from current confidence from the smaller gain from confidence history.

## 6. Related Work

I'm not proposing new theory here, I'm mainly combining a few existing ideas:

- **Model calibration.** The idea that a model's predicted probability should roughly match how often it's actually correct. My ECE and reliability diagram calculations follow the standard approach used in calibration research (Guo et al., 2017).
- **Using confidence to catch misclassifications.** Using a model's own softmax confidence to flag likely wrong predictions is a known baseline approach (Hendrycks & Gimpel, 2017).
- **Robustness under input noise.** Checking how a model's output changes as the input gets noisier or harder is a common way to study reliability.

What I did differently is combine these: instead of looking at a single confidence value, I look at how confidence behaves across a sequence of increasingly corrupted versions of the same input, and use that sequence to train a detector.

## 7. Dataset

I used `sklearn.datasets.fetch_20newsgroups`, which is public and well documented.

- **Task:** Binary text classification between `alt.atheism` and `soc.religion.christian`. I picked these two categories on purpose because they share a lot of overlapping vocabulary (both discuss religion), so the model doesn't just get every prediction right with 99% confidence. This gives more realistic confidence behavior to study.
- **Preprocessing:** I removed headers, footers, and quoted text from each post so the model can't just cheat off metadata.
- **Size:** 1,754 documents in total. I split this into 1,315 for training and 439 for testing (stratified 75/25 split). The classes are fairly balanced, 44.4% vs 55.6%.
- **Ground truth:** the actual newsgroup each post came from, which is a clear and measurable label.

## 8. Methodology

### 8.1 Base classifier

I used TF-IDF (unigrams and bigrams, up to 20,000 features) feeding into a Logistic Regression classifier. This is the model whose confidence I study throughout the rest of the project. I report its own performance separately from the drift detector's performance, since the brief asked me to keep these two clearly apart.

### 8.2 Generating confidence sequences

One problem I ran into early on: the dataset doesn't naturally give you repeated predictions on the same thing, each document only appears once. But the whole point of this project is to look at how confidence changes across a sequence of predictions. So instead of making up numbers, I generated a real sequence for every test document by corrupting it at six increasing levels (0%, 10%, 20%, 30%, 40%, 50% of the words removed and the rest shuffled) and running the classifier on each corrupted version.

This gave me 2,634 real predictions in total (439 documents times 6 steps each), and every single confidence value came from an actual `predict_proba()` call. I kept the corruption seeds deterministic (based on a hash of the document ID) so that re-running the whole pipeline gives the exact same results every time.

I want to be clear about one thing: the true label doesn't change as I corrupt the text. I'm studying how the same document behaves as its content gets noisier, not relabeling anything.

### 8.3 Feature engineering

Covered in section 10 below.

### 8.4 Confidence drift detector

I trained a second classifier to predict `unreliable` (1 for an incorrect prediction). Experiments A, B, and C add feature families in sequence: input-corruption context, current confidence, and then confidence history. Logistic Regression is the primary comparison model. I also evaluated Random Forest and Gradient Boosting on C.

### 8.5 Train/test split

I split the data at the document level using `GroupShuffleSplit` (75/25), so that different corruption steps from the same document never appear in both training and test data. I used that holdout split for model details and a separate five-fold `GroupKFold` evaluation for the main A/B/C comparison. Each row receives one out-of-fold prediction. The saved model's algorithm was selected using three grouped folds inside the training portion of the holdout split; the holdout labels did not choose it.

For the high-confidence review policy, I made five-fold out-of-fold predictions on the *base classifier's 1,315 training documents*. A predicted class enters the review rule if it produced at least one high-confidence error there. If no such error is observed in a policy-selection set, the conservative fallback reviews all high-confidence predicted classes. I checked this rule with nested cross-validation on those training documents and then applied the fitted rule to the separate 439-document base-model test set. The follow-up policy was designed after the detector failure was observed, so the test-set result is a retrospective check, not independent prospective confirmation.

### 8.6 Original-error detector after six probes

The per-step C detector has no history at step 0, so it cannot answer whether the original prediction was wrong using drift. I built a second detector whose target is precisely that original error. For each of the 1,315 base-training documents, I generated six confidence readings from a base classifier trained on other documents in a stratified fold. I summarized the complete trajectory, its class switches, original confidence and predicted class, and document length into one row per document. The error detector is a balanced Logistic Regression model. Its review threshold is 75% of the lowest score assigned to a high-confidence training error in five meta-model training folds. This conservative margin was chosen during development; it favors recall at the cost of false alarms. No test-set label is used to fit the model or compute its threshold.

At inference, `score_documents` in `src/sequence_error_detector.py` produces the six probes from unlabeled text, computes the same features, and returns the original classification's error-risk score and flag. The flag is available only *after* those probes, not at the initial prediction. I assessed the procedure in five outer folds over the base-training documents, rebuilding the base classifier, features, detector, and threshold without each outer validation document. Finally I evaluated the saved detector on the separate 439-document base-model test set. This detector was developed after inspecting that test set's earlier failure, so the latter result is exploratory rather than prospective confirmation.

## 9. ML Model

| Component | Algorithms I tried |
|---|---|
| Base text classifier | Logistic Regression on TF-IDF features |
| Confidence drift detector | Logistic Regression (baseline), Random Forest, Gradient Boosting |
| Original-error sequence detector | Logistic Regression on completed six-step sequences |

The `unreliable` target is imbalanced (only about 21% of predictions are actually wrong), so I used `class_weight="balanced"` for Logistic Regression and Random Forest, and equivalent sample weighting for Gradient Boosting (which doesn't support `class_weight` directly). Without this, the models just predicted "reliable" for almost everything and F1 collapsed close to 0, which wasn't a useful result to report.

## 10. Feature Engineering

The primary C model uses 13 inputs: two context features, two current-confidence features, and nine history features. Every primary input is available by the current prediction step. Ground truth is kept only as the training target and for evaluation.

| Feature | What it means |
|---|---|
| `confidence` | confidence at the current step |
| `prev_confidence` | confidence at the previous step |
| `confidence_diff` | current minus previous |
| `confidence_pct_change` | percentage change from the previous step |
| `moving_avg_confidence` | average confidence so far |
| `moving_std_confidence` | standard deviation of confidence so far |
| `rolling_std_confidence` | standard deviation over the last 3 steps (a volatility measure) |
| `confidence_range_so_far` | highest minus lowest confidence seen so far |
| `rate_of_change` | change in confidence divided by the step number |
| `consecutive_decreases` | how many steps in a row confidence has been dropping |
| `prediction_margin` | how far apart the two class probabilities are |
| `step`, `corruption_level` | where we are in the sequence |

I tested the feature groups by removing change features, moving statistics, and consecutive drops in turn. I also tested a class-level historical accuracy feature separately. That optional statistic uses labels from training documents only, after the document split; it never enters the primary C comparison. In this binary task, `prediction_margin` is a deterministic transform of confidence, so it does not add independent information. See section 13 for the ablation results.

## 11. Confidence vs Actual Correctness

This is really the core comparison of the project. I binned the step 0 predictions (the uncorrupted, real deployment predictions) by confidence and checked accuracy in each bin.

| Confidence range | Number of predictions | Accuracy |
|---|---|---|
| 0-20% | 0 | not applicable |
| 20-40% | 0 | not applicable |
| 40-60% | 156 | 63.5% |
| 60-80% | 223 | 90.1% |
| 80-100% | 60 | 93.3% |

No predictions fall below 50%, because in a two class softmax, the winning class always gets at least 50%. Accuracy does go up as confidence goes up, so confidence isn't meaningless, but it's not perfectly monotonic either (I get into this more in section 12).

**Trend across the 439 six step sequences:**
- Average confidence started at 0.663 and ended at 0.633.
- 72.0% of documents showed an overall drop in confidence as corruption increased. 27.3% actually went up (sometimes removing words makes a text look more clearly like one class, oddly enough). The rest stayed about the same.
- Of 439 documents, 25 moved from correct at the start to incorrect at the final step. Net confidence change had a correlation of 0.090 with that transition. Its correlation with *any* incorrect final prediction (95 documents, including those wrong at the start) was 0.266. These are different outcomes and must not be conflated.

The sign is also informative: a simple rule that treats every decrease as a warning would not describe these data well. The sequence's overall change is a weak signal for a correct prediction becoming wrong, while the final-error outcome has a larger association. Neither correlation establishes a useful decision threshold on its own.

## 12. Calibration Analysis

I calculated this on the step 0 (uncorrupted) test predictions.

- **Expected Calibration Error (ECE): 0.153**
- **Binary Brier score: 0.160**, calculated from the probability of class 1 and the observed 0/1 label

| Confidence bin | n | Average confidence | Average accuracy |
|---|---|---|---|
| 0.5 to 0.6 | 156 | 0.555 | 63.5% |
| 0.6 to 0.7 | 135 | 0.648 | 86.7% |
| 0.7 to 0.8 | 88 | 0.745 | 95.5% |
| 0.8 to 0.9 | 50 | 0.850 | 96.0% |
| 0.9 to 1.0 | 10 | 0.920 | 80.0% |

The reliability diagram is saved at `results/figures/reliability_diagram.png`. What I found interesting is that the model is actually a bit under confident in the 0.6 to 0.9 range (its accuracy is higher than its stated confidence there). But the very top bin, 0.9 to 1.0, actually has lower accuracy than the bin below it. That bin only has 10 samples though, so it could just be noise from a small sample size. Either way it's a good reminder that "very high confidence" is not a guarantee of correctness, which is exactly the motivation for section 13.

## 13. Experiments

Each experiment predicts whether the same base-model prediction is incorrect. A uses `step` and `corruption_level`; B adds current confidence and margin; C adds nine history features to B. The primary comparison uses pooled out-of-fold predictions from five document-grouped folds across all 2,634 rows. No document crosses a fold boundary. Logistic Regression and the 0.5 decision threshold are the same for A, B, and C.

| Experiment | Added information | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| A: context only | corruption step and level | 0.504 | 0.296 | 0.509 |
| B: current confidence | confidence and margin, in addition to A | 0.637 | 0.468 | 0.729 |
| C: confidence history | nine drift features, in addition to B | 0.688 | **0.508** | **0.763** |

The C-minus-B gain was 0.040 in F1 and 0.034 in ROC-AUC. A paired bootstrap of 1,000 document resamples gave intervals of 0.018 to 0.065 for F1 and 0.014 to 0.053 for ROC-AUC. Those intervals treat the trained fold models as fixed; they do not include variation from retraining or a new dataset. The five fold scores and pooled metrics are in `results/metrics/drift_history_validation.json`.

On the separate 110-document holdout, Logistic Regression scored F1 0.253 / 0.447 / 0.457 and ROC-AUC 0.508 / 0.741 / 0.764 for A / B / C. This single split shows a smaller C-minus-B gain than the pooled five-fold result. Random Forest and Gradient Boosting on C reached holdout F1 0.472 and 0.464, respectively. Three-fold grouped cross-validation on the training documents chose Logistic Regression as the saved model. The full confusion matrices and error rates are in `results/metrics/drift_detector_experiments.json`; model selection is in `results/metrics/algorithm_selection.json`.

`results/figures/experiment_comparison.png` plots this holdout comparison; its bars are not the pooled five-fold scores in the table above.

Removing the moving-statistics group from C lowered holdout ROC-AUC from 0.764 to 0.751. Removing the change group gave 0.763, and removing consecutive drops gave 0.764. These group ablations suggest that moving statistics contributed the clearest incremental signal on this split. A class-accuracy prior computed only from training documents raised holdout F1 to 0.483 and ROC-AUC to 0.787, but it is a separate class prior, not evidence about drift history. Full results are in `results/metrics/feature_ablation.json`.

## 14. Results

- The base classifier does its actual job well: 81.1% accuracy and 0.892 ROC-AUC on held out data (full breakdown in section 15).
- Confidence is informative on average (section 11), but it's not perfectly calibrated (section 12).
- Net drift alone correlates 0.090 with a correct prediction becoming wrong. It correlates 0.266 with a wrong prediction at the final step, a broader outcome.
- With context and current confidence held constant, nine history features improved pooled out-of-fold F1 from 0.468 to 0.508. The fixed holdout gain was smaller, from 0.447 to 0.457.
- Six of 100 high-confidence predictions were wrong, accounting for 7.2% of the uncorrupted base model's errors. The per-step detector missed all six at step 0. The new sequence-level detector flagged all six after six probes and flagged 50 correct predictions. The separate immediate review policy routed all six and 78 correct predictions to review.

## 15. Error Analysis

**Base model performance** (step 0, held out test set, kept separate from the drift detector as required):

- Accuracy: 0.811, Precision: 0.804, Recall: 0.873, F1: 0.837, ROC-AUC: 0.892
- Confusion matrix: rows are actual class, columns are predicted class, class 0 is alt.atheism and class 1 is soc.religion.christian:

```
[[143, 52],
 [31, 213]]
```

**High confidence errors** (confidence of 75% or higher):
- 100 out of 439 step 0 predictions (22.8%) were high confidence.
- 6 of those 100 (6.0%) were wrong. One example: the model was 96.3% confident and still got it wrong.
- Interestingly, all 6 high confidence errors went the same direction: the true label was `alt.atheism` but the model predicted `soc.religion.christian`. My guess is that this happens because a lot of atheism related posts quote or reference Christian scripture and terminology while arguing against it, and a TF-IDF plus bigram model can't really tell the difference between someone quoting scripture to argue against it and someone writing as a believer. This is a reasonable explanation based on what I can see in the data, but I haven't proven it's the actual cause.
- I tested detection directly. Across five document-grouped folds, every uncorrupted prediction received an out-of-fold C-model result. Among the 100 high-confidence predictions, six were wrong; the detector flagged none of those six and raised no false alarms in this subset at its 0.5 threshold (`results/metrics/high_confidence_detector_analysis.json`). At step 0, history features have not yet accumulated. These six cases are too few for a precise recall estimate, but the observed miss is a clear failure for the motivating use case.

I added a **separate review policy** for this failure mode. Five-fold out-of-fold base-model predictions on the 1,315 *training* documents showed six high-confidence errors, all predicted as class 1. The saved policy therefore routes predictions at 75% confidence or higher in that predicted class to human review. No label from the 439-document base-model test set chooses the class or threshold. Applying the policy to those test predictions routed all six errors, but also 78 correct predictions: 84 reviews among 100 high-confidence cases and only 7.1% precision in the review queue. A nested five-outer/three-inner-fold check on the training corpus routed all six observed high-confidence errors and 225 correct predictions; when an inner fold observes no high-confidence errors, the policy conservatively reviews all predicted classes. The rule is intentionally broad. It does not predict which individual reviews are wrong, and this follow-up was designed after inspecting the test-set failure. It needs confirmation on a new corpus before deployment. The saved rule and full counts are in `models/high_confidence_review_policy.json` and `results/metrics/high_confidence_review_analysis.json`.

The **sequence-level original-error detector** uses the completed six-probe trajectory instead of trying to predict the original error at step 0. On the 439-document test set, it flagged all 6 high-confidence errors and 50 of the 94 correct high-confidence predictions, for 56 alerts and 10.7% alert precision. Its high-confidence ROC-AUC was 0.794. A start-only model with original confidence, predicted class, and document length flagged 5/6 with 43 false alarms; adding the sequence features caught one more error but increased the review load by seven. In a five-fold nested validation on the base-training corpus, the complete procedure flagged all six observed high-confidence errors and 171 correct predictions. Those results are in `results/metrics/sequence_error_detector_analysis.json`. The primary saved model is `models/drift_detector_best.joblib`; `models/drift_detector_per_step.joblib` preserves the earlier A/B/C model. These are small counts, the threshold favors recall, and the design followed inspection of the test errors. I would not treat 6/6 here as a guarantee on unseen data.

## 16. Limitations

- **The sequences are simulated, not naturally repeated.** I generated confidence drift by corrupting the same document repeatedly, not by collecting genuinely independent repeated queries. I did this on purpose so that every confidence value would be real, but it doesn't capture other real world sources of drift, like the model being updated over time or the input distribution shifting.
- **The dataset is fairly small.** 439 test documents (2,634 total sequence rows) isn't huge, and some of my bins, like the 0.9 to 1.0 calibration bin with only 10 samples, are too small to draw strong conclusions from on their own.
- **Only two classes.** I don't know if these results carry over to multi class problems.
- **The label is the source newsgroup, not a judgment of the text's stance.** Several `alt.atheism` posts contain long Christian passages or republished messages. After removing headers and quoted metadata, the remaining text may not identify which newsgroup published it. High confidence in the wrong source label is therefore not always a recognizable anomaly in the model's available input.
- **Only one type of base model.** I only tested TF-IDF plus Logistic Regression. Other model types, especially deep neural networks, tend to be overconfident in different ways, so the calibration story could look different there.
- **The original per-step detector's performance is still modest.** Pooled out-of-fold F1 was 0.508 for C, and the separate holdout F1 was 0.457 for Logistic Regression. It missed all six high-confidence errors at the uncorrupted step. The sequence-level detector covers them only after six probes and flags 50 correct predictions. The immediate review policy flags 78 correct predictions.
- **The document level split reduces training data for the detector on purpose.** I used 1,974 rows from 329 documents rather than all 2,634 rows, to avoid leaking information across the split. This is the correct thing to do, but it does mean the detector has less to learn from.
- **The evidence comes from one base model and one dataset.** The bootstrap intervals measure variation from sampling these documents with the fold models fixed. A new corpus or retrained base model could change the size or direction of the history-feature gain.

## 17. Future Scope

- Try the same approach on a different kind of base model, like a neural text classifier, to see if confidence drift features are still useful when the model's calibration behaves differently.
- If I ever get access to genuinely repeated real world queries (the same input asked multiple times over time) instead of synthetic corruption, I'd like to try this on that kind of data.
- Try calibration techniques like temperature scaling or Platt scaling on the base model, and check whether that changes how much value the drift features add.
- Extend this to multi class classification, where confidence and margin features would be richer (for example, using entropy over the full probability distribution instead of just the top class).
- Reduce the sequence-level detector's false alarms and test whether its six-probe method works beyond this corpus. For immediate step-0 coverage, the review policy remains costly.

## 18. Conclusion

Confidence generally tracked correctness in this dataset: accuracy rose from 63.5% in the 40-60% confidence bin to 93.3% in the 80-100% bin. Calibration was imperfect (ECE 0.153; binary Brier score 0.160), and six high-confidence predictions were wrong. Net confidence change had little association with a correct prediction becoming incorrect (r = 0.090), though its association with any incorrect final prediction was larger (r = 0.266).

The matched experiments show that confidence history adds a modest signal beyond current confidence on these corrupted-text sequences. Five-fold pooled F1 rose from 0.468 to 0.508 and ROC-AUC from 0.729 to 0.763. The separate holdout gain was smaller. The per-step detector missed all six high-confidence errors at the uncorrupted step. The sequence-level detector, trained to predict that original error after six probes, flagged all six and 50 correct predictions in this retrospective test. That improves on the immediate class-based review policy's 78 correct reviews, but it is still expensive and cannot warn at step 0. A new dataset is needed to test whether the result holds.

## 19. References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR.
- Lang, K. (1995). *NewsWeeder: Learning to Filter Netnews* (the original source of the 20 Newsgroups dataset).
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825 to 2830.
