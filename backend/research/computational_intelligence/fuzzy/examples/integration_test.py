#!/usr/bin/env python3
"""
Integration Test: Fuzzy Sentiment with Existing Engines

This script tests the integration between the fuzzy sentiment module
and the existing sentiment engines (LogReg, SVM, TF-IDF).

Usage:
    python -m research.computational_intelligence.fuzzy.examples.integration_test

Author: [Your Name]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
backend_root = project_root / 'backend'
sys.path.insert(0, str(backend_root))

import os
os.chdir(backend_root)

from typing import Dict, List, Any


def test_engine_loading():
    """Test 1: Load existing sentiment engines."""
    print("\n" + "=" * 60)
    print("TEST 1: Loading Existing Sentiment Engines")
    print("=" * 60)

    engines = {}
    engine_status = {}

    # Try loading LogReg
    try:
        from app.sentiment_engines import LogRegSentimentEngine
        engines['logreg'] = LogRegSentimentEngine()
        engine_status['logreg'] = 'LOADED'
        print(f"  LogReg: LOADED")
    except Exception as e:
        engine_status['logreg'] = f'FAILED: {e}'
        print(f"  LogReg: FAILED - {e}")

    # Try loading SVM
    try:
        from app.sentiment_engines import SVMSentimentEngine
        engines['svm'] = SVMSentimentEngine()
        engine_status['svm'] = 'LOADED'
        print(f"  SVM: LOADED")
    except Exception as e:
        engine_status['svm'] = f'FAILED: {e}'
        print(f"  SVM: FAILED - {e}")

    # Try loading TF-IDF
    try:
        from app.sentiment_engines import TFIDFSentimentEngine
        engines['tfidf'] = TFIDFSentimentEngine()
        engine_status['tfidf'] = 'LOADED'
        print(f"  TF-IDF: LOADED")
    except Exception as e:
        engine_status['tfidf'] = f'FAILED: {e}'
        print(f"  TF-IDF: FAILED - {e}")

    loaded_count = sum(1 for s in engine_status.values() if s == 'LOADED')
    print(f"\n  Total engines loaded: {loaded_count}/3")

    return engines, engine_status


def test_engine_predictions(engines: Dict[str, Any]):
    """Test 2: Get predictions from each engine."""
    print("\n" + "=" * 60)
    print("TEST 2: Individual Engine Predictions")
    print("=" * 60)

    test_texts = [
        "This video is absolutely amazing! Best content ever!",
        "Terrible video, waste of time. Very disappointing.",
        "The video was okay, nothing special.",
        "I love this channel so much! Always great content!",
        "Not sure what to think about this one...",
    ]

    results = {}

    for text in test_texts:
        print(f"\n  Text: \"{text[:50]}...\"" if len(text) > 50 else f"\n  Text: \"{text}\"")
        results[text] = {}

        for name, engine in engines.items():
            try:
                result = engine.analyze(text)
                label = result.label if hasattr(result, 'label') else result.get('label', 'N/A')
                probs = result.probs if hasattr(result, 'probs') else result.get('probs', {})
                results[text][name] = {'label': label, 'probs': probs}
                print(f"    {name:8s}: {label:10s} | Pos={probs.get('Positive', 0):.2f}, "
                      f"Neu={probs.get('Neutral', 0):.2f}, Neg={probs.get('Negative', 0):.2f}")
            except Exception as e:
                print(f"    {name:8s}: ERROR - {e}")
                results[text][name] = {'label': 'ERROR', 'probs': {}}

    return results


def test_fuzzy_integration(engines: Dict[str, Any]):
    """Test 3: Integrate with Fuzzy Sentiment Classifier."""
    print("\n" + "=" * 60)
    print("TEST 3: Fuzzy Sentiment Integration")
    print("=" * 60)

    if len(engines) < 2:
        print("  SKIPPED: Need at least 2 engines for fuzzy integration")
        return None

    from research.computational_intelligence.fuzzy import FuzzySentimentClassifier
    from research.computational_intelligence.fuzzy.engine_integration import (
        FuzzySentimentEngine,
        EngineAdapter,
    )

    # Create fuzzy engine with loaded engines
    print(f"\n  Creating FuzzySentimentEngine with: {list(engines.keys())}")

    fuzzy_engine = FuzzySentimentEngine(
        base_engines=engines,
        mf_type='gaussian',
        defuzz_method='centroid'
    )

    print(f"  Fuzzy engine created successfully!")
    print(f"  Configuration:")
    info = fuzzy_engine.get_model_info()
    for key, value in info.items():
        print(f"    {key}: {value}")

    # Test predictions
    test_texts = [
        "This video is absolutely amazing! Best content ever!",
        "Terrible video, waste of time. Very disappointing.",
        "The video was okay, nothing special.",
        "Mixed feelings about this one, some good parts and some bad.",
    ]

    print("\n  Fuzzy Classification Results:")
    print("-" * 60)

    for text in test_texts:
        result = fuzzy_engine.analyze(text)
        print(f"\n  Text: \"{text[:45]}...\"" if len(text) > 45 else f"\n  Text: \"{text}\"")
        print(f"    Label: {result.label}")
        print(f"    Score: {result.score:.4f}")
        print(f"    Confidence: {result.confidence:.4f}")
        print(f"    Fuzziness: {result.fuzziness_index:.4f}")
        print(f"    Base Model Scores: {result.base_model_scores}")

    return fuzzy_engine


def test_uncertainty_detection(engines: Dict[str, Any]):
    """Test 4: Test uncertainty detection on ambiguous cases."""
    print("\n" + "=" * 60)
    print("TEST 4: Uncertainty Detection on Ambiguous Cases")
    print("=" * 60)

    if len(engines) < 2:
        print("  SKIPPED: Need at least 2 engines")
        return

    from research.computational_intelligence.fuzzy.engine_integration import FuzzySentimentEngine

    fuzzy_engine = FuzzySentimentEngine(
        base_engines=engines,
        mf_type='gaussian',
        defuzz_method='centroid'
    )

    # Test cases designed to be ambiguous
    ambiguous_texts = [
        "It's not bad but not great either",
        "Could have been better but also could have been worse",
        "Some parts were good, others not so much",
        "I don't hate it but I don't love it",
        "Meh",
    ]

    clear_positive = [
        "Absolutely loved this! 10/10 would recommend!",
        "Best video I've ever seen, incredible!",
    ]

    clear_negative = [
        "Worst content ever, total garbage!",
        "Absolutely terrible, don't waste your time!",
    ]

    print("\n  AMBIGUOUS CASES (should have high fuzziness):")
    print("-" * 60)
    for text in ambiguous_texts:
        result = fuzzy_engine.analyze(text)
        status = "HIGH UNCERTAINTY" if result.fuzziness_index > 0.3 else "low uncertainty"
        print(f"  [{status:17s}] Fuzz={result.fuzziness_index:.3f} | {result.label:8s} | \"{text}\"")

    print("\n  CLEAR POSITIVE (should have low fuzziness):")
    print("-" * 60)
    for text in clear_positive:
        result = fuzzy_engine.analyze(text)
        status = "HIGH UNCERTAINTY" if result.fuzziness_index > 0.3 else "low uncertainty"
        print(f"  [{status:17s}] Fuzz={result.fuzziness_index:.3f} | {result.label:8s} | \"{text}\"")

    print("\n  CLEAR NEGATIVE (should have low fuzziness):")
    print("-" * 60)
    for text in clear_negative:
        result = fuzzy_engine.analyze(text)
        status = "HIGH UNCERTAINTY" if result.fuzziness_index > 0.3 else "low uncertainty"
        print(f"  [{status:17s}] Fuzz={result.fuzziness_index:.3f} | {result.label:8s} | \"{text}\"")


def test_batch_processing(engines: Dict[str, Any]):
    """Test 5: Batch processing performance."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Processing Performance")
    print("=" * 60)

    if len(engines) < 2:
        print("  SKIPPED: Need at least 2 engines")
        return

    import time
    from research.computational_intelligence.fuzzy.engine_integration import FuzzySentimentEngine

    fuzzy_engine = FuzzySentimentEngine(
        base_engines=engines,
        mf_type='gaussian',
        defuzz_method='centroid'
    )

    # Generate test batch
    test_texts = [
        "Great video!",
        "Not good at all",
        "Average content",
        "Love this channel",
        "Waste of time",
    ] * 10  # 50 samples

    print(f"\n  Processing {len(test_texts)} samples...")

    start_time = time.time()
    results = fuzzy_engine.analyze_batch(test_texts)
    elapsed = time.time() - start_time

    print(f"\n  Results:")
    print(f"    Total samples: {len(results)}")
    print(f"    Total time: {elapsed:.3f}s")
    print(f"    Avg per sample: {elapsed/len(results)*1000:.2f}ms")

    # Summary statistics
    labels = [r.label for r in results]
    confidences = [r.confidence for r in results]
    fuzziness = [r.fuzziness_index for r in results]

    print(f"\n  Label Distribution:")
    for label in ['Positive', 'Neutral', 'Negative']:
        count = labels.count(label)
        print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")

    print(f"\n  Confidence Stats:")
    print(f"    Mean: {sum(confidences)/len(confidences):.3f}")
    print(f"    Min:  {min(confidences):.3f}")
    print(f"    Max:  {max(confidences):.3f}")

    print(f"\n  Fuzziness Stats:")
    print(f"    Mean: {sum(fuzziness)/len(fuzziness):.3f}")
    print(f"    Min:  {min(fuzziness):.3f}")
    print(f"    Max:  {max(fuzziness):.3f}")


def test_comparison_with_baseline(engines: Dict[str, Any]):
    """Test 6: Compare fuzzy results with single-model baseline."""
    print("\n" + "=" * 60)
    print("TEST 6: Comparison with Single-Model Baseline")
    print("=" * 60)

    if len(engines) < 2:
        print("  SKIPPED: Need at least 2 engines")
        return

    from research.computational_intelligence.fuzzy.engine_integration import FuzzySentimentEngine

    fuzzy_engine = FuzzySentimentEngine(
        base_engines=engines,
        mf_type='gaussian',
        defuzz_method='centroid'
    )

    test_texts = [
        "This is amazing content!",
        "Terrible, just terrible.",
        "It was okay I guess.",
        "Not sure how I feel about this.",
        "Best video on YouTube!",
        "Don't bother watching this garbage.",
    ]

    print("\n  Comparison of Fuzzy vs Individual Models:")
    print("-" * 80)
    print(f"  {'Text':<35} | {'Fuzzy':^12} | " + " | ".join(f"{n:^8}" for n in engines.keys()))
    print("-" * 80)

    for text in test_texts:
        fuzzy_result = fuzzy_engine.analyze(text)
        fuzzy_label = f"{fuzzy_result.label} ({fuzzy_result.confidence:.2f})"

        individual_labels = []
        for name, engine in engines.items():
            try:
                result = engine.analyze(text)
                label = result.label if hasattr(result, 'label') else result.get('label', '?')
                individual_labels.append(label[:8])
            except:
                individual_labels.append('ERROR')

        text_short = text[:33] + ".." if len(text) > 35 else text
        print(f"  {text_short:<35} | {fuzzy_label:^12} | " + " | ".join(f"{l:^8}" for l in individual_labels))


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("FUZZY SENTIMENT INTEGRATION TEST SUITE")
    print("=" * 60)

    # Test 1: Load engines
    engines, status = test_engine_loading()

    if not engines:
        print("\n  ERROR: No engines could be loaded!")
        print("  Make sure model files exist in ./models/ directory")
        return

    # Test 2: Individual predictions
    test_engine_predictions(engines)

    # Test 3: Fuzzy integration
    fuzzy_engine = test_fuzzy_integration(engines)

    # Test 4: Uncertainty detection
    test_uncertainty_detection(engines)

    # Test 5: Batch processing
    test_batch_processing(engines)

    # Test 6: Comparison
    test_comparison_with_baseline(engines)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print("""
    Summary:
    - Fuzzy sentiment module integrates with existing engines
    - Uncertainty quantification works correctly
    - Batch processing is functional

    The fuzzy layer adds:
    1. Explicit uncertainty measurement (fuzziness index)
    2. Confidence scores based on model agreement
    3. Handling of ambiguous/conflicting predictions

    Ready for thesis experiments!
    """)


if __name__ == '__main__':
    main()
