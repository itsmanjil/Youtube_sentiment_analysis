# Error Characterization (LogReg baseline)

Model: `model.sav`  |  Test set: `test.csv`  |  n = 165,110  |  overall accuracy = **0.6946**

> This file complements `error_analysis.md` (which shows individual high-confidence misclassifications). It slices the same errors by **text properties** (length, negation, question-like) so that the thesis Limitations section can point to concrete, quantified failure modes instead of isolated examples.

## Per-class performance

| Class | n | Accuracy | Recall | Precision |
|---|---:|---:|---:|---:|
| Negative | 59,614 | 0.7242 | 0.7242 | 0.6914 |
| Neutral | 50,540 | 0.6232 | 0.6232 | 0.6172 |
| Positive | 54,956 | 0.7283 | 0.7283 | 0.7751 |

## Accuracy by text length

| Length bucket | n | Accuracy |
|---|---:|---:|
| very_short (<=5 tok) | 15,781 | 0.7335 |
| short (6-15) | 73,083 | 0.6976 |
| medium (16-40) | 59,044 | 0.6872 |
| long (41+) | 17,202 | 0.6717 |

## Effect of negation markers

| Slice | n | Accuracy |
|---|---:|---:|
| with_negation | 46,791 | 0.6567 |
| without_negation | 118,319 | 0.7096 |

**Δ (with - without negation): -0.0529.** A negative delta indicates negation tokens make classification harder, which supports the choice in `src/preprocessing/classical.py` to keep negators in the stopword list and run negation tagging.

## Question-like vs statement-like

| Slice | n | Accuracy |
|---|---:|---:|
| question_like | 16,176 | 0.7264 |
| statement_like | 148,934 | 0.6912 |

## Confidence distribution for correct vs wrong predictions

| Outcome | Mean confidence | Std |
|---|---:|---:|
| Correct | 0.7436 | 0.1752 |
| Wrong   | 0.5933  | 0.1431 |

Confidence gap (correct − wrong) = **+0.1503**. A small gap means the model is 'confidently wrong' too often — exactly the calibration weakness the neuro-fuzzy gate targets.

## Most common confusion pairs

| True → Pred | Count | % of errors |
|---|---:|---:|
| Neutral → Negative | 12,120 | 24.0% |
| Negative → Neutral | 11,755 | 23.3% |
| Positive → Neutral | 7,782 | 15.4% |
| Positive → Negative | 7,151 | 14.2% |
| Neutral → Positive | 6,925 | 13.7% |
| Negative → Positive | 4,687 | 9.3% |

## Most confidently-wrong examples

| # | True | Pred | Conf | Len | Neg? | Text (first 200 chars) |
|---:|---|---|---:|---|---|---|
| 1 | Neutral | Positive | 0.999 | very_short (<=5 tok) | no | tera thank you |
| 2 | Neutral | Positive | 0.999 | very_short (<=5 tok) | no | thank you carryminati |
| 3 | Neutral | Positive | 0.999 | very_short (<=5 tok) | no | love kaitlan collins red heart |
| 4 | Neutral | Positive | 0.999 | medium (16-40) | no | heartfelt thanks to amazing ai for rescuingsaving the digital cheetah her adorable cute baby cheetahs are spectacularly precious sight its beautifully heartwarming smiling face with hearts |
| 5 | Negative | Positive | 0.998 | very_short (<=5 tok) | no | great shes pilferer red heart |
| 6 | Neutral | Positive | 0.998 | short (6-15) | no | learned lot through telusuko thank you |
| 7 | Neutral | Positive | 0.997 | medium (16-40) | yes | god bless you mel cant wait for the resurrection movie was absolutely devastated when watched your first one the passion sat there and just cried the whole time thank you red heart |
| 8 | Positive | Negative | 0.997 | short (6-15) | no | what disgusting cover up by disgusting people |
| 9 | Neutral | Positive | 0.997 | short (6-15) | no | love this mr gibson red heart |
| 10 | Positive | Negative | 0.997 | medium (16-40) | no | ridiculous worried about being called racist while little girls being raped in the worst way ridiculous |
| 11 | Neutral | Positive | 0.997 | very_short (<=5 tok) | no | thank you truckers |
| 12 | Positive | Negative | 0.997 | short (6-15) | no | please americans just dump trump the past lying toxic propaganda pushing fool |
| 13 | Neutral | Positive | 0.996 | medium (16-40) | no | congratulations im so glad this channel is growing so well great to see channel get the recognition they deserve |
| 14 | Neutral | Negative | 0.996 | very_short (<=5 tok) | yes | ugh cant stand awkward silences |
| 15 | Neutral | Positive | 0.996 | medium (16-40) | no | laura you are the bomb this interview nearly broke my heart in transcendental way thank you mel for your service to truth bravo smiling face with sunglasses heart on fire |
| 16 | Negative | Positive | 0.996 | short (6-15) | no | thank you for helping this amazing ia you are the future |
| 17 | Neutral | Positive | 0.996 | short (6-15) | no | love this guy the best actor and filmmaker in the world god bless mel gibson |
| 18 | Negative | Positive | 0.996 | long (41+) | yes | whenever saw your videos lost my mind and get stuck just by thinking whether may have programming skills like you or not most of the time really get disappointed because as software engineering studen |
| 19 | Neutral | Positive | 0.996 | medium (16-40) | no | babymonster red heart im so happy yall stan them they are my fav red heart red heart |
| 20 | Neutral | Positive | 0.995 | medium (16-40) | no | sir wish could understand hindicoz see your videos very informative and miss alot of details wish could have translations thank you thank you |

## Thesis-ready takeaways

1. **Length matters.** Accuracy on *long (41+)* comments is 0.6717 vs 0.7335 on *very_short (<=5 tok)* — short comments lose information that TF-IDF needs to discriminate classes.
2. **Negation is a weak spot.** Accuracy changes by -0.0529 when a negation marker is present. This validates the thesis choice to include negation tagging in the classical preprocessing path.
3. **Dominant confusion pair: Neutral → Negative (24.0% of all errors).** Addressing this single confusion would deliver the biggest headline improvement.
4. **Confidence gap = +0.1503.** The model is only modestly more confident when it is right than when it is wrong, which is exactly why calibration (not accuracy) is the honest contribution of the CI layer.
