# Thesis Revision Summary

**Input:** Thesis_YouTube_Sentiment_Final.docx
**Output:** Thesis_YouTube_Sentiment_Revised.docx
**Formatting reference:** sample.pdf

All data, results, figures, tables, citations, and claims are unchanged. Only wording, grammar, flow, and formatting were revised.

## 1. Structural fixes

- **Restored the missing "Chapter 3B: System Design and Implementation" heading.** It was glued onto the end of the Label-Noise Discussion paragraph as plain text, so the chapter had no heading and was absent from the TOC. It is now a proper Heading 1.
- **Merged 109 hard-wrapped line fragments in Chapter 3B into real paragraphs.** Sections 3B.1–3B.9 had each line as a separate paragraph (a paste artifact), which broke justification and spacing.
- **Converted literal "•" lines into proper Word bullet lists** (meta-learner, neuro-fuzzy gate, temperature scaling, and deployment platform sections), matching the list style used elsewhere in the document.

## 2. Language and clarity edits

- Fixed grammar and punctuation errors, e.g. missing colon in "Many deployment objectives conflict: maximising macro-F1 can worsen calibration"; missing comma in "PSO vs NSGA-II ensemble weighting, stacked meta-learning, and neuro-fuzzy gating".
- Expanded contractions for academic tone (isn't → is not, doesn't → does not, it's → it is).
- Replaced conversational or overly rhetorical phrasing with plain academic wording:
  - "Think of brand monitoring..." → "In practical settings such as brand monitoring..."
  - "the honest comparator... weak strawman" → "the fair comparison point... a weak one"
  - "Common thesis-killers" → "Common evaluation errors"
  - "A finding worth stating directly rather than leaving implicit" → "One finding should be stated directly"
  - "an honest limitation, not a claimed win" → "reported as a limitation rather than a positive finding"
  - "One oddity: 'face'..." → "One unusual entry is 'face'..."
- Smoothed long, comma-spliced sentences in the Abstract, Chapter 2, 3B.3, 3B.6 (deployment note), and Chapter 4 (calibration discussion) by splitting them into shorter sentences. All numbers and claims kept verbatim.
- Removed stray backticks around code identifiers in the meta-learner section.

## 3. Formatting changes (matched to sample.pdf)

- Heading 1: Cambria 16 pt bold, black (was blue Word default).
- Heading 2: Cambria 14 pt bold, black (was blue).
- Captions: Cambria 11 pt, black (was small grey).
- Body: line spacing set to 1.15 on body paragraphs (sample uses ~1.15 leading; body font was already Cambria 12 pt, matching the sample).
- Title page: centred and set to black (was justified, blue).
- Page setup already matched the sample (Letter, 1-inch margins, "Page | N" footer) and was left as is.
- All 21 figures, 18 tables, the TOC field, and the List of Figures were preserved untouched.

## 4. Image placeholders added (6)

1. `[Insert image: workflow diagram – data preprocessing and splitting pipeline]` — Chapter 3, after Table 2.
2. `[Insert image: code screenshot – blind annotation tool (scripts/annotate.py)]` — Chapter 3, Gold-Set section.
3. `[Insert image: code screenshot – base classifier training configuration]` — Section 3B.2.
4. `[Insert image: system screenshot – analyst dashboard (result overview)]` — Section 3B.8.
5. `[Insert image: system screenshot – live analysis monitoring page]` — Section 3B.8.
6. `[Insert image: output/result screenshot – gold-set evaluation console output]` — Appendix A.

## 5. Remaining issues to check manually

- **Update fields in Word:** open the document, select all (Ctrl+A), press F9 to refresh the Table of Contents and List of Figures — page numbers shifted and the new Chapter 3B heading must be picked up. Then check the List of Figures entries still match.
- **Title page:** the sample thesis has no institution name/logo or date on the title page reference I could see; add your university name, logo, and submission date per your department template.
- **Empty "Headline Metrics" section:** the System Overview's Headline Metrics heading is followed only by a table; confirm this is intentional.
- **Numbered list in Section 5.4** (the three NSGA-II objectives) uses literal "1./2./3." text, kept as-is; convert to a Word numbered list if you prefer.
- **Replace the six placeholders** with real screenshots, then caption them as numbered figures so they appear in the List of Figures.
- **Declaration/acknowledgements pages:** the sample-style front matter (declaration, acknowledgements) is absent; add if your programme requires it.

## 6. Chapter renumbering (second pass)

Chapters were renumbered to a plain sequence, with all in-text cross-references, section numbers, captions, and List of Figures entries updated to match:

| Old | New |
|---|---|
| Chapter 3 (continued): Exploratory Data Analysis | Chapter 4: Exploratory Data Analysis |
| Chapter 3B: System Design and Implementation (3B.1–3B.9) | Chapter 5: System Design and Implementation (5.1–5.9) |
| Chapter 4: Consolidated Evaluation | Chapter 6: Consolidated Evaluation |
| Chapter 5: Conclusion and Future Work | Chapter 7: Conclusion and Future Work |
| Threats to Validity | Chapter 8: Threats to Validity |

Cross-reference updates include "Chapter 3B" → "Chapter 5", "§3B.x" → "§5.x", "Chapter 3B.1" → "Section 5.1", "Section 4.6" → "Section 6.6", and evaluation/future-work references ("Chapter 4" → "Chapter 6", "Chapter 5" → "Chapter 7"). References to methodology as "Chapter 3" were left unchanged. Remember to refresh the TOC/List of Figures (Ctrl+A, F9).
