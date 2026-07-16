# Thesis Copy-Edit Change Log

File: `Thesis_YouTube_Sentiment_CopyEdited.docx` (revised from `Thesis_YouTube_Sentiment_Final.docx`).
No claim, result, number, statistic, citation, equation, model name, dataset name, or metric value was altered. All figures and tables are retained with their original content.

## (a) Clarity and flow edits

**Chapter 1**
- "The claim here is narrower, and more defensible" → removed the unnecessary comma.
- "Think of brand monitoring…" → "Consider brand monitoring…"; "isn't enough" → "is not enough"; "it's an incomplete measure" → "it is an incomplete measure" (contractions removed for academic register).
- RQ1, RQ3, and RQ4 labels bolded through the topic phrase so all four research questions match RQ2's pattern.
- Contributions list: added the missing comma in "PSO vs NSGA-II ensemble weighting, stacked meta-learning, and neuro-fuzzy gating".
- "Labels are automated, not human origin" → "not of human origin".

**Chapter 2**
- Trailing "though" reworked twice: "…corpora on which much early work was validated, though." → "However, social-media sentiment differs materially…"; "The present thesis is explicit, though, that…" → "However, the present thesis is explicit that…".
- "Second, noise and informality, typos, elongations…, inflate vocabulary" → parenthesised the example list so the sentence parses cleanly.
- "weak strawman" → "weak straw man".
- Route A terminology note: hyphen-as-dash replaced with a colon; "Route A the research agenda remains future work… route_a_live_v1 the runtime artifact is complete and is what produces…" → proper appositive commas and "…is complete and produces…".
- Fixed missing punctuation: "Many deployment objectives conflict maximising macro-F1 can worsen calibration" → "…objectives conflict: maximising…".
- Temperature-scaling sentence: nested appositions parenthesised ("temperature scaling (dividing the logits by a single learned scalar T before the softmax)").

**Chapter 3 / EDA**
- Fixed comma splice in the 'face' token discussion ("…rather than organic vocabulary: convert-mode emoji processing renders emoji such as…").
- "confirm this doesn't materially change" → "confirm that this does not materially change".
- "The author's pass isn't fully arms-length the way a third-party-only annotation would be, which is a limitation, but…" → "…is not fully arms-length in the way…; this is a limitation, but…".

**Chapter 3B**
- Meta-learner inference sentence split in two ("…final class probabilities. This is a learned, data-driven combination rule…").
- "Both, though, route through…" → "Both, however, route through…".
- "Should base_models not match…" → "If base_models does not match…".
- Long fuzzy-gate closing sentence split ("…in this study (Chapter 4). It is a second, independently derived route…").

**Chapter 4**
- The two long all-bold notes ("Note on the fuzzy ensemble row (revised)" and "Why McNemar and the macro-F1 bootstrap can disagree") are now regular text with a bold lead-in only.
- "Statistical testing backs this up" → "Statistical testing supports this result".
- "Two naming caveats apply. Here, ensemble denotes… And fuzzy_ensemble is omitted" → "First, ensemble here denotes… Second, fuzzy_ensemble is omitted".
- "The headline pattern: abstaining…" → "The headline pattern is that abstaining…".
- "line up with" → "are consistent with" (Table 6 comparison) and "The key conclusions line up with Guo et al. (2017)" → "…align with Guo et al. (2017)".
- Dangling verb fixed: "among single models, logistic regression is (ECE 0.0039)" → "…is the best calibrated (ECE 0.0039)".
- "The fairest reading of NSGA-II's contribution in this codebase: it recovers…" → "…is that it recovers…".
- "Figure references below plot…" → "Figure 17 plots…".
- "agree with the automated source labels 73.5%" → "…73.5% of the time".
- "The two models are tied on accuracy, then, but…" → "The two models are therefore tied on accuracy, but…".

**Chapter 5 / Threats to Validity**
- "ABSA. And while artifact pinning…" → "ABSA. In addition, while artifact pinning…".
- "Common thesis-killers" → "Common evaluation pitfalls".
- "Mitigations: no hyperparameter was selected on the test set - hyperparameters…" → "Mitigations are in place: …test set; hyperparameters…".
- "this is disclosed deliberate reuse" → "this is a disclosed, deliberate reuse".
- "A residual gap remains, though:" → "However, a residual gap remains:"; "The principal mitigation, though, is" → "The principal mitigation, however, is".

## (b) Consistency and terminology standardizations

1. **macro-F1** — hyphenated everywhere in prose and captions ("Per-class and macro F1" caption and the "Macro F1" table header fixed). Table column headers keep the capitalised "Macro-F1" form.
2. **artifact** — "artefact" (2 occurrences) → "artifact", matching the dominant spelling. Bibliography titles untouched.
3. **Naive Bayes** — "TF-IDF + Naive-Bayes base classifier" de-hyphenated to match all other uses.
4. **Section references** — "§3B.x" (5 occurrences) → "Section 3B.x" / "Sections 3B.3–3B.4" to match the "Chapter X" cross-reference style used everywhere else.
5. **Krippendorff's α** — the symbol α is now used whenever a numeric value follows ("Krippendorff's α = 0.9547"; also "Cohen's/Fleiss' κ = 0.9546"); the spelled-out "alpha" is retained in purely narrative mentions ("using Krippendorff's alpha to quantify…"). Table 13's "Krippendorff α" → "Krippendorff's α".
6. **British spelling** — "transformer-centered" (2) → "transformer-centred", matching optimisation/normalised/labelled/behaviour used throughout. The American spelling in the Kennedy & Eberhart bibliography title was left as published.
7. **Numeric ranges** — hyphens replaced with en dashes: 0.6630–0.6970, 0.6622–0.6959, 0.66–0.70, 0.81–0.85 (matching existing en-dash ranges).
8. **Multiplication sign** — "1.35x" → "1.35×".
9. **"10k"** → "10,000-row" (domain split description).
10. Rules confirmed already consistent (no change needed): "vs" without a period; code identifiers (logreg, svm, tfidf, ensemble_pso, ensemble_nsga2, fuzzy_ensemble, meta_learner, hybrid_dl, route_a_live_v1) kept verbatim; "gold set" as noun, "gold-set" as modifier; Positive/Neutral/Negative capitalised as class names.

## (c) Formatting changes made to match the sample thesis

(Applied in this editing round and the immediately preceding one; listed here as the complete deviation report.)

- **Body font**: document was largely Arial with dark-grey (#1A1A1A) text → Cambria 12 pt black throughout, matching the sample's body text.
- **Headings**: blue (#1F3864/#2E5496) with an underline rule → plain bold black, 16 pt (chapter/H1) and 14 pt (section/H2), matching the sample's Cambria bold black headings; the sample has no coloured or ruled headings.
- **Line spacing**: three different spacings were mixed (1.04, 1.15, 1.25) → normalised to ~1.15 throughout, matching the sample's measured line pitch.
- **Figure captions**: small grey italic → regular black, centred beneath each figure, "Figure N Title" — the sample's caption style. Bold caption labels removed. Figures are kept on the same page as their captions.
- **Margins/page**: already matched the sample (US Letter, 1-inch margins, justified body, "Page | N" footer with numbering restarting at Chapter 1) — no change needed.
- **Chapter flow**: every chapter-level heading now starts on a new page, as in the sample. "Chapter 3B: System Design and Implementation" was not a real heading (it was embedded inside a body paragraph and missing from the TOC) — converted to a proper Heading 1.
- **Chapter 3B text**: had been pasted as one-line-per-paragraph fragments (ragged right edge, wrong spacing) — 116 fragments merged back into proper justified paragraphs.
- **Tables**: cell text normalised to a consistent 11 pt (was 9 pt with one inconsistent row in Table 10).
- **Keywords section** added after the Abstract (the sample has one) — see [CHECK] item below.
- **Title page**: centred and set in black to read cleanly (the sample's cover is an institutional image page that cannot be replicated; see [CHECK]).
- **TOC and List of Figures**: refreshed so page numbers and new sections (Chapter 3B, Keywords, Appendix B) appear correctly.

## (d) [CHECK] items that need your judgment

- **[CHECK: chapter numbering]** "Chapter 3 (continued): Exploratory Data Analysis" and "Chapter 3B" are unconventional. Renumbering them (e.g. EDA → Chapter 4, System Design → Chapter 5, and shifting later chapters) would be cleaner but touches dozens of cross-references ("Chapter 3B", "Section 3B.6", figure captions), so I did not do it. Tell me if you want the full renumbering.
- **[CHECK: keywords]** I drafted the Keywords line (YouTube comments; sentiment analysis; ensemble learning; particle swarm optimisation; NSGA-II; neuro-fuzzy systems; probability calibration; selective prediction; reproducibility) — confirm or replace with your programme's required keywords.
- **[CHECK: gold-set figure wording]** The gold-set evaluation text referred to "the full-300-item figures" while Table 10 reports 291 reconciled labels (9 disputed items excluded). I softened this to "the full gold-set figures" — confirm this reads as you intend.
- **[CHECK: abstract length]** The Abstract is ~3 dense paragraphs and includes detailed bug-audit history. Examiners often expect ~1 page maximum; consider trimming the bug-list detail (I did not cut any content).
- **[CHECK: title page]** The sample's cover is an institutional/university cover page image. Yours is text-only; if your programme provides an official cover template, it should replace the current title page.
- **[CHECK: TOC scope]** Your TOC lists front matter (System Overview, Abstract, Keywords); the sample's TOC starts at the Introduction. I kept yours — remove the front-matter entries if your programme requires it.
- **[CHECK: fields]** After inserting the screenshots, select all (Ctrl+A) and press F9 in Word (or right-click the TOC → Update Field) to refresh the TOC, List of Figures, and page numbers.

## (e) Image placeholders added

All placeholders use the exact `[Insert image: …]` format, shown in bold red so they are easy to find. No decorative images were added; your 21 existing workflow/architecture figures already cover the diagram needs.

**Section 3B.8 (Deployment Platform)** — 2 placeholders:
- [Insert image: dashboard results screenshot]
- [Insert image: search page screenshot]

**Appendix B: Screenshots** (new appendix, mirroring the sample's Screenshots appendix) — 12 placeholders:
- Application: [Insert image: login page screenshot], [Insert image: dashboard screenshot], [Insert image: monitoring page screenshot], [Insert image: report page screenshot]
- Code: [Insert image: API endpoint code screenshot], [Insert image: PSO code screenshot], [Insert image: NSGA-II code screenshot], [Insert image: neuro-fuzzy gate code screenshot], [Insert image: calibration code screenshot]
- Output: [Insert image: evaluation console output screenshot], [Insert image: API JSON response screenshot], [Insert image: annotation tool screenshot]
