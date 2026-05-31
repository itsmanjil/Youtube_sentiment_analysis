# Chapter 2 — Literature Review

Status date: 2026-05-31
Length: ~2,600 words

This chapter situates the thesis within five bodies of work: (i) sentiment
analysis and social-media text classification; (ii) classical machine-learning
baselines; (iii) transformer-based sentiment models; (iv) computational
intelligence for model combination — particle swarm optimisation, evolutionary
multi-objective optimisation, ensemble learning, and fuzzy / neuro-fuzzy
systems; and (v) calibration, uncertainty quantification, and selective
prediction. It closes with human annotation and inter-annotator agreement.

## 2.1 Sentiment Analysis and Social-Media Text

Sentiment analysis — the computational treatment of opinion, polarity, and
affect in text — has matured from lexicon-driven polarity scoring into a core
supervised NLP task (Pang & Lee, 2008; Liu, 2012). Early systems relied on
sentiment lexicons and hand-built rules; supervised statistical classifiers
then became dominant as labelled corpora grew. Social-media sentiment, however,
differs materially from the review-style corpora on which much early work was
validated. User-generated platform text is short, noisy, and laden with
non-standard orthography, emoji, slang, hashtags, code-switching, and sarcasm
(Barbieri et al., 2020; Rosenthal et al., 2017). YouTube comments are a
particularly demanding instance: they are brief (in the corpus used here the
median comment is ~14 words), highly topical (sentiment-bearing vocabulary is
domain-dependent), and frequently *neutral* or *off-topic* in ways that resist
clean polarity assignment.

These properties create three recurring difficulties. First, **short text**
gives each instance little lexical context, so classifiers must generalise from
sparse evidence (Song et al., 2014). Second, **noise and informality** — typos,
elongations ("loooove"), emoji, and platform-specific tokens — inflate
vocabulary and require careful normalisation. Third, **sarcasm and
context-dependence** mean surface lexical cues can invert the true polarity
(Joshi et al., 2017). A fourth issue, **class ambiguity**, is central to this
thesis: the Neutral class is the least separable, because it is defined
residually (neither clearly positive nor negative) and overlaps with both poles.
Empirically, the exploratory analysis in Chapter 3 shows Neutral comments are
the *shortest* in the corpus (median 12 words versus 16 for Negative and 15 for
Positive), compounding the short-text problem precisely where the label is most
ambiguous.

## 2.2 Classical Machine-Learning Baselines

Despite the rise of deep learning, classical linear models remain strong,
efficient, and interpretable baselines for text classification, and are widely
used as the comparison floor in sentiment studies. The canonical pipeline pairs
**TF-IDF** feature representation (Sparck Jones, 1972; Salton & Buckley, 1988) —
which weights terms by frequency within a document against their rarity across
the corpus — with a linear classifier such as **logistic regression** or a
**linear support vector machine** (Joachims, 1998; Wang & Manning, 2012). Wang
and Manning (2012) showed that well-tuned linear SVM and naive-Bayes-SVM
variants over n-gram TF-IDF features are remarkably competitive with more
complex models on sentiment tasks, a result that continues to hold for short,
high-volume social text where transformer inference cost is prohibitive.

These baselines matter for the present work for two reasons. First, they are the
honest comparator: any computational-intelligence ensemble must demonstrate
value *over* a tuned logistic-regression baseline, not merely over a weak
strawman. Second, linear models over sparse features are exceptionally
inexpensive at inference (sub-millisecond per comment in this system), which is
material for a deployable YouTube-scale pipeline. The thesis therefore retains
logistic regression, linear SVM, and TF-IDF as first-class baselines rather than
discarding them, and reports the (small) margin by which ensemble and
meta-learning methods exceed them.

## 2.3 Transformer-Based Sentiment Models

The transformer architecture (Vaswani et al., 2017) and the bidirectional
pre-training paradigm of **BERT** (Devlin et al., 2019) redefined the state of
the art across NLP, including sentiment classification. By pre-training deep
bidirectional encoders on large unlabelled corpora and fine-tuning on
task-specific data, these models capture contextual meaning that bag-of-words
representations cannot. Successors refined the recipe: **RoBERTa** (Liu et al.,
2019) improved pre-training robustness, and **DeBERTa** (He et al., 2021)
introduced disentangled attention and an enhanced mask decoder, yielding strong
results on language-understanding benchmarks.

Crucially for social media, **domain-adapted** transformers outperform
general-purpose ones on platform text. Barbieri et al. (2020) introduced
TweetEval and a RoBERTa model pre-trained on a large Twitter corpus,
demonstrating that in-domain pre-training substantially improves sentiment and
emotion classification on noisy social text. This motivates the *Route A*
encoder direction in this project (DeBERTa-v3 as an English encoder baseline).
However, the present thesis is explicit that the transformer work here is
**not** a completed full-dataset evaluation: only a smoke/CPU benchmark exists,
the classification head of the saved checkpoint is not fully trained, and the
deployed headline is classical-and-ensemble-first. Treating the encoder line as
future work — rather than over-claiming transformer superiority — is a
deliberate threat-to-validity decision (see Chapter 4 limitations).

## 2.4 Computational Intelligence for Model Combination

The methodological core of this thesis is *how to combine* base classifiers, and
this is where computational intelligence enters. Ensemble learning — combining
multiple learners to outperform any single member — is well established
(Dietterich, 2000): bagging, boosting, and stacking reduce variance and/or bias
by aggregating diverse models. The open question this work addresses is **how to
choose the combination weights** when the objective is not only accuracy but
also calibration.

**Particle Swarm Optimisation (PSO).** PSO (Kennedy & Eberhart, 1995) is a
population-based metaheuristic in which candidate solutions ("particles") move
through a search space guided by their own best position and the swarm's best
position. It is well suited to continuous weight-optimisation problems such as
finding ensemble mixing weights, requiring no gradient information. In this
thesis PSO is used to optimise ensemble weights for a *single* objective
(validation F1), producing the `ensemble_pso` configuration.

**Multi-objective optimisation with NSGA-II.** Many deployment objectives
conflict: maximising macro-F1 can worsen calibration, and vice versa. The
Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002) is the canonical
evolutionary algorithm for such problems. NSGA-II maintains a population, ranks
solutions by Pareto dominance, and uses crowding distance to preserve diversity
along the Pareto front, returning a set of non-dominated trade-off solutions
rather than a single point. Applying NSGA-II to ensemble weighting allows the
thesis to expose the F1-versus-calibration trade-off explicitly and to select a
*knee-point* configuration (`ensemble_nsga2`) that balances both — directly
operationalising RQ1.

**Stacked meta-learning.** Stacking (Wolpert, 1992) trains a meta-learner on the
out-of-fold predictions of base models, learning a data-driven combination
rather than fixed weights. In this work the stacked meta-learner is the strongest
model by macro-F1, and is compared against the metaheuristic-weighted ensembles.

**Fuzzy and neuro-fuzzy systems.** Fuzzy logic (Zadeh, 1965) represents partial
truth via membership functions, providing a principled way to reason under
linguistic uncertainty — appropriate for sentiment, where membership in
"Positive" is rarely all-or-nothing. Neuro-fuzzy systems such as ANFIS (Jang,
1993) combine fuzzy inference with neural-network parameter learning. This
thesis uses a neuro-fuzzy *gate* that fuzzifies base-model confidences and
applies fuzzy inference to modulate the ensemble decision under uncertainty.
The honest empirical finding (Chapter 4) is that the fuzzy ensemble is a valid,
implemented computational-intelligence component but does not improve the
full-test headline metrics — a result the thesis reports rather than conceals.

## 2.5 Calibration, Uncertainty, and Selective Prediction

A central argument of this thesis is that **probability quality**, not only
label accuracy, determines the usefulness of a deployed sentiment system. Three
strands of literature support this.

**Calibration.** A classifier is calibrated if its predicted confidence matches
its empirical accuracy. Guo et al. (2017) showed that modern neural networks,
despite high accuracy, are frequently *mis*-calibrated (typically
over-confident), and that a simple post-hoc method — **temperature scaling**,
dividing the logits by a single learned scalar T before the softmax —
substantially restores calibration without changing the predicted label.
Calibration quality is quantified by **Expected Calibration Error (ECE)**, which
bins predictions by confidence and averages the gap between confidence and
accuracy, and by the **Brier score** (Brier, 1950), a proper scoring rule
measuring the mean squared error between predicted probabilities and one-hot
outcomes. This thesis adopts temperature scaling per model, reports ECE and
Brier for every model, and explicitly refuses to claim universal calibration
gains — the evidence shows calibration improvements are model-specific.

**Uncertainty quantification.** Beyond point calibration, predictive uncertainty
can be estimated and exploited. Lakshminarayanan et al. (2017) showed that
*deep ensembles* — averaging multiple independently trained models — yield
well-calibrated predictive uncertainty that is simple and competitive with
Bayesian approaches. This provides theoretical grounding for the ensemble
methods at the heart of this work: combining diverse base learners is not only
an accuracy device but an uncertainty-estimation device. Normalised predictive
**entropy** is used here as the per-comment uncertainty signal.

**Selective prediction.** When a model can abstain on uncertain inputs, it can
trade *coverage* for *accuracy*. The classifier-with-reject-option framework
(Chow, 1970; El-Yaniv & Wiener, 2010) formalises this, and Geifman and El-Yaniv
(2017) developed selective prediction for deep networks, characterising the
risk–coverage curve. This thesis builds entropy-gated selective prediction and
reports coverage–accuracy curves per model (RQ2): a model whose confidence is
genuinely informative will show steeply rising accuracy as low-confidence
comments are abstained upon. This reframes "weak" Neutral performance
constructively — the system can flag ambiguous Neutral comments for human review
rather than forcing an error.

## 2.6 Human Annotation and Inter-Annotator Agreement

Because the source corpus is automatically (not human) labelled, the reliability
of the evaluation depends on an independent human reference. Measuring the
reliability of categorical annotation requires chance-corrected agreement
coefficients rather than raw percentage agreement, which is inflated by the
base rate of the majority class. **Cohen's kappa** (Cohen, 1960) corrects for
chance agreement between two annotators; **Fleiss' kappa** generalises this to
multiple annotators; and **Krippendorff's alpha** (Krippendorff, 2004) provides
a general reliability coefficient applicable to any number of annotators,
missing data, and different measurement levels. Artstein and Poesio (2008)
review these measures for computational linguistics and discuss thresholds for
"tentative" (α ≥ 0.67) and "reliable" (α ≥ 0.80) conclusions; Landis and Koch
(1977) provide widely cited interpretive bands for kappa. This thesis
constructs a 300-item gold set, has it independently labelled by two annotators,
and reports Krippendorff's α = 0.9547 (strong agreement), using the reconciled
human labels to validate the headline metrics and to separate label error from
model error (RQ4).

## 2.7 Research Gap and Positioning

The reviewed literature establishes mature components in isolation: classical
baselines, transformers, PSO and NSGA-II, fuzzy systems, calibration, selective
prediction, and agreement metrics. What remains comparatively under-explored — and
what this thesis addresses — is their **integrated, honestly-evaluated
combination for social-media sentiment under deployment constraints**:
specifically, using *multi-objective* evolutionary optimisation to weight an
ensemble for the *joint* F1–calibration objective, wiring the result into an
*artifact-pinned* runtime whose deployed predictions provably match the offline
benchmark, and grounding every claim in a re-runnable artifact and an
independent human gold set. The contribution is therefore one of rigorous
integration and evaluation methodology rather than a novel single algorithm.

## References

- Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics*, 34(4), 555–596.
- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of EMNLP 2020*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
- Dietterich, T. G. (2000). Ensemble methods in machine learning. *Multiple Classifier Systems*, LNCS 1857, 1–15.
- El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective classification. *JMLR*, 11, 1605–1641.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 2017*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
- He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR 2021*.
- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
- Joachims, T. (1998). Text categorization with support vector machines. *ECML 1998*.
- Joshi, A., Bhattacharyya, P., & Carman, M. J. (2017). Automatic sarcasm detection: A survey. *ACM Computing Surveys*, 50(5), 1–22.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*, 1942–1948.
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2nd ed.). Sage.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
- Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool.
- Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
- Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in IR*, 2(1–2), 1–135.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). SemEval-2017 Task 4: Sentiment analysis in Twitter. *SemEval 2017*.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.
- Song, G., Ye, Y., Du, X., Huang, X., & Bie, S. (2014). Short text classification: A survey. *Journal of Multimedia*, 9(5).
- Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *ACL 2012*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
