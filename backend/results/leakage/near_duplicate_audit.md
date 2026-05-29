# Near-Duplicate Leakage Audit

- Created at: `2026-05-17T07:59:27Z`
- Split directory: `C:\Users\itsma\OneDrive\Documents\GitHub\Youtube_sentiment_analysis\backend\data\route_a_benchmark_cpu`
- Text column: `text`
- Records scanned: `810`
- Candidate pairs checked: `7302`
- Exact cross-split duplicates: `0`
- Near-duplicate cross-split pairs: `9`
- Status: `REVIEW`

## Interpretation

Potential cross-split near duplicates were found. Review the examples, tighten deduplication, or report the residual risk as a limitation.

## Near-Duplicate Examples

- `val:28` vs `test:30` (Hamming distance 1)
  - A: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy red heart red heart red heart
  - B: face with tears of joy face with tears of joy face with tears of joy
- `train:248` vs `val:38` (Hamming distance 2)
  - A: bro called a dog instead of that bird lol face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy
  - B: Umm no sorry just the dog for pictures face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy
- `train:248` vs `test:30` (Hamming distance 2)
  - A: bro called a dog instead of that bird lol face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy
  - B: face with tears of joy face with tears of joy face with tears of joy
- `train:273` vs `test:30` (Hamming distance 2)
  - A: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy “skip the cheapest person I know”
  - B: face with tears of joy face with tears of joy face with tears of joy
- `val:38` vs `test:30` (Hamming distance 2)
  - A: Umm no sorry just the dog for pictures face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy
  - B: face with tears of joy face with tears of joy face with tears of joy
- `train:273` vs `val:120` (Hamming distance 3)
  - A: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy “skip the cheapest person I know”
  - B: they seem real fruity face with tears of joy face with tears of joy face with tears of joy face with tears of joy
- `val:120` vs `test:30` (Hamming distance 3)
  - A: they seem real fruity face with tears of joy face with tears of joy face with tears of joy face with tears of joy
  - B: face with tears of joy face with tears of joy face with tears of joy
- `train:248` vs `val:28` (Hamming distance 3)
  - A: bro called a dog instead of that bird lol face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy
  - B: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy red heart red heart red heart
- `train:273` vs `val:28` (Hamming distance 3)
  - A: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy “skip the cheapest person I know”
  - B: face with tears of joy face with tears of joy face with tears of joy face with tears of joy face with tears of joy red heart red heart red heart
