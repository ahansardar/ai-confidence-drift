# AI Confidence Drift Detection Using Machine Learning

**Organization:** PranavX Labs
**Internship Domain:** Artificial Intelligence & Machine Learning
**Project Type:** AI/ML Research + Experimental Implementation

---

## 1. Abstract

AI models usually give a confidence score along with their prediction, but that confidence doesn't always match how correct the model actually is. In this project I wanted to check if changes in a model's confidence, tracked across a series of related predictions, can tell us something useful about whether a prediction should be trusted.

I trained a simple text classifier (TF-IDF + Logistic Regression) on a public two class newsgroup dataset. To actually get a "series" of confidence values for each sample, I fed every test document to the model multiple times, each time with a bit more of the text corrupted. This gave me real confidence sequences, not made up numbers. From these sequences I built statistical features (moving average, volatility, how many times confidence dropped in a row, and so on) and used them to train a second model whose job is to guess if a prediction is unreliable.

The results were mixed in an interesting way. Just looking at how much a prediction's confidence changed overall was a weak signal (correlation of about 0.04 with the prediction ending up wrong). But when I used the full set of drift features, the detector got noticeably better at flagging unreliable predictions: F1 went from 0.25 with no confidence information at all, up to 0.48 with the full feature set, and ROC-AUC went from 0.51 to 0.79. I also checked calibration and found the base model is only okay, not great (ECE of 0.153), and 6% of its high confidence predictions (above 75%) were still wrong, including one case where the model was 96% confident and completely wrong.

So overall, confidence drift does carry useful information, but only if you look at the shape of the sequence rather than just the total change from start to end.

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

I tested this instead of just assuming it. As it turns out, part of the hypothesis holds and part of it doesn't, which I explain in section 14.

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

I trained a second, separate classifier to predict a binary target I call `unreliable` (1 if the prediction was wrong, 0 if it was correct), using only the engineered features. I compared three different feature sets (Experiments A, B, and C, explained in section 13) and three algorithms: Logistic Regression, Random Forest, and Gradient Boosting.

### 8.5 Train/test split

I split the data at the document level using `GroupShuffleSplit` (75/25), so that different corruption steps from the same document never end up on both sides of the split. If I hadn't done this, the detector could basically cheat by seeing a slightly less corrupted version of the same document in training and the test version in testing.

## 9. ML Model

| Component | Algorithms I tried |
|---|---|
| Base text classifier | Logistic Regression on TF-IDF features |
| Confidence drift detector | Logistic Regression (baseline), Random Forest, Gradient Boosting |

The `unreliable` target is imbalanced (only about 21% of predictions are actually wrong), so I used `class_weight="balanced"` for Logistic Regression and Random Forest, and equivalent sample weighting for Gradient Boosting (which doesn't support `class_weight` directly). Without this, the models just predicted "reliable" for almost everything and F1 collapsed close to 0, which wasn't a useful result to report.

## 10. Feature Engineering

I didn't just throw every possible feature at the model. I picked features that only use information available up to the current step, nothing from future steps and nothing from the true label, since that would be cheating.

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
| `historical_accuracy_for_class` | how accurate the model has historically been for this predicted class |
| `step`, `corruption_level` | where we are in the sequence |

I didn't just assume these were all useful, that's exactly what Experiments A, B, and C in section 13 are for.

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
- Correlation between "total confidence drift" and "the prediction ended up wrong at the last step" was only 0.043. That's very weak.

This last point matters a lot. The brief specifically warns not to assume a falling confidence trend means the model is getting less accurate, and I actually tested that assumption instead of taking it for granted. It turned out the assumption was basically wrong, at least for net drift on its own.

## 12. Calibration Analysis

I calculated this on the step 0 (uncorrupted) test predictions.

- **Expected Calibration Error (ECE): 0.153**
- **Brier score: 0.125**

| Confidence bin | n | Average confidence | Average accuracy |
|---|---|---|---|
| 0.5 to 0.6 | 156 | 0.555 | 63.5% |
| 0.6 to 0.7 | 135 | 0.648 | 86.7% |
| 0.7 to 0.8 | 88 | 0.745 | 95.5% |
| 0.8 to 0.9 | 50 | 0.850 | 96.0% |
| 0.9 to 1.0 | 10 | 0.920 | 80.0% |

The reliability diagram is saved at `results/figures/reliability_diagram.png`. What I found interesting is that the model is actually a bit under confident in the 0.6 to 0.9 range (its accuracy is higher than its stated confidence there). But the very top bin, 0.9 to 1.0, actually has lower accuracy than the bin below it. That bin only has 10 samples though, so it could just be noise from a small sample size. Either way it's a good reminder that "very high confidence" is not a guarantee of correctness, which is exactly the motivation for section 13.

## 13. Experiments

I ran a controlled comparison using the same train/test split every time (1,974 training rows from 329 documents, 660 test rows from 110 documents). The goal in every experiment is the same: predict whether a prediction is unreliable.

| Experiment | Features used | Algorithm | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| A: no confidence features | just `step` and `corruption_level` | Logistic Regression | 0.498 | 0.253 | 0.508 |
| B: single point confidence | `confidence`, `prediction_margin` | Logistic Regression | 0.661 | 0.443 | 0.741 |
| C: full confidence drift features | all 13 engineered features | Logistic Regression | 0.705 | **0.483** | **0.787** |
| C: full confidence drift features | all 13 engineered features | Random Forest | 0.726 | 0.472 | 0.752 |
| C: full confidence drift features | all 13 engineered features | Gradient Boosting | 0.726 | 0.471 | 0.762 |

Full numbers, including confusion matrices and false positive/negative rates, are saved in `results/metrics/drift_detector_experiments.json`.

What this tells me is that confidence drift features genuinely help. Going from experiment A to experiment C, F1 improved by 0.230 and ROC-AUC improved by 0.279. That's a real, measured improvement, not something I'm assuming. That said, I want to be honest that F1 of 0.48 still isn't great in absolute terms, this isn't a solved problem, it's a meaningful step forward.

## 14. Results

- The base classifier does its actual job well: 81.1% accuracy and 0.892 ROC-AUC on held out data (full breakdown in section 15).
- Confidence is informative on average (section 11), but it's not perfectly calibrated (section 12).
- Net confidence drift alone is a poor predictor of whether a prediction ends up wrong (correlation of just 0.043). This means my original hypothesis, in its simple form, doesn't really hold.
- However, the shape of the confidence sequence, things like volatility, moving average, and consecutive drops, does carry real predictive signal. So a more precise version of my hypothesis does hold: it's not about how much confidence changed overall, it's about the pattern of how it changed.
- 6.0% of high confidence (75% or above) predictions were wrong, and those account for 7.2% of all the errors I saw.

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
- These are exactly the kind of cases my drift detector is meant to catch. The detector gets a ROC-AUC of 0.79 overall at flagging unreliable predictions, but that doesn't mean it catches every single high confidence error specifically. I want to be upfront that this is a real limitation, not something I'm claiming to have fully solved.

## 16. Limitations

- **The sequences are simulated, not naturally repeated.** I generated confidence drift by corrupting the same document repeatedly, not by collecting genuinely independent repeated queries. I did this on purpose so that every confidence value would be real, but it doesn't capture other real world sources of drift, like the model being updated over time or the input distribution shifting.
- **The dataset is fairly small.** 439 test documents (2,634 total sequence rows) isn't huge, and some of my bins, like the 0.9 to 1.0 calibration bin with only 10 samples, are too small to draw strong conclusions from on their own.
- **Only two classes.** I don't know if these results carry over to multi class problems.
- **Only one type of base model.** I only tested TF-IDF plus Logistic Regression. Other model types, especially deep neural networks, tend to be overconfident in different ways, so the calibration story could look different there.
- **The detector's performance is still modest.** An F1 of about 0.48 for the best confidence drift detector means it still misses a lot of unreliable predictions, or flags some reliable ones incorrectly. This is a real result and an improvement, but not a finished solution.
- **The document level split reduces training data for the detector on purpose.** I used 1,974 rows from 329 documents rather than all 2,634 rows, to avoid leaking information across the split. This is the correct thing to do, but it does mean the detector has less to learn from.

## 17. Future Scope

- Try the same approach on a different kind of base model, like a neural text classifier, to see if confidence drift features are still useful when the model's calibration behaves differently.
- If I ever get access to genuinely repeated real world queries (the same input asked multiple times over time) instead of synthetic corruption, I'd like to try this on that kind of data.
- Try calibration techniques like temperature scaling or Platt scaling on the base model, and check whether that changes how much value the drift features add.
- Extend this to multi class classification, where confidence and margin features would be richer (for example, using entropy over the full probability distribution instead of just the top class).

## 18. Conclusion

Confidence, on average, does line up with correctness in this project, accuracy rose from about 63% to about 93% as I moved across confidence bins. But it's not fully calibrated (ECE of 0.153), and it's not something you can trust blindly either, 6% of high confidence predictions were still wrong. The raw amount that confidence changed across a sequence was a weak predictor of error on its own (correlation around 0.04). But a model trained on the shape of the confidence sequence, its volatility, its moving statistics, its consecutive drops, did give a real and measurable improvement at flagging unreliable predictions, taking F1 from 0.25 up to 0.48 and ROC-AUC from 0.51 up to 0.79.

So my answer to the research question is yes, but with an important condition that I tested rather than assumed: confidence drift is useful when you look at its statistical shape, not just at whether it went up or down overall. And even the best version of my detector here still leaves real room to improve, which I think is an honest place to end this project.

## 19. References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR.
- Lang, K. (1995). *NewsWeeder: Learning to Filter Netnews* (the original source of the 20 Newsgroups dataset).
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825 to 2830.
