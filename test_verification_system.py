#!/usr/bin/env python3
"""
test_verification_system.py

Test suite for the multi-stage verification system.
Validates accuracy on known test cases.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple

try:
    from matcher_verification import VerifiedMatcher
    from manual_review_ui import ManualReviewUI
except ImportError as e:
    print(f"Error: Could not import verification modules: {e}")
    print("Make sure matcher_verification.py and manual_review_ui.py are in the same directory")
    exit(1)

class VerificationTester:
    """Test suite for verification system."""
    
    def __init__(self, index_dir="faiss_index", canonical_dir="canonical"):
        print("Initializing verification tester...")
        self.matcher = VerifiedMatcher(index_dir=index_dir, canonical_dir=canonical_dir)
        self.test_results = []
    
    def load_ground_truth(self, ground_truth_file: str) -> List[Dict]:
        """
        Load ground truth test cases.
        
        Format:
        [
            {
                "image_path": "test_data/card_001.png",
                "true_id": "12345",
                "true_name": "Blue-Eyes White Dragon",
                "difficulty": "easy"  # easy, medium, hard
            },
            ...
        ]
        """
        if not Path(ground_truth_file).exists():
            print(f"Ground truth file not found: {ground_truth_file}")
            return []
        
        with open(ground_truth_file, 'r') as f:
            return json.load(f)
    
    def test_single_card(self, image_path: str, true_id: str, true_name: str) -> Dict:
        """Test a single card and return results."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "image_path": image_path,
                "error": "Could not load image",
                "success": False
            }
        
        # Run verification
        start_time = time.time()
        result = self.matcher.match_with_verification(img, confidence_threshold=0.95)
        elapsed = time.time() - start_time
        
        # Check if correct
        predicted_id = result.get('card_id')
        correct = (predicted_id == true_id)
        
        return {
            "image_path": image_path,
            "true_id": true_id,
            "true_name": true_name,
            "predicted_id": predicted_id,
            "predicted_name": result.get('filename'),
            "correct": correct,
            "verified": result.get('verified', False),
            "confidence": result.get('confidence', 0.0),
            "scores": result.get('scores', {}),
            "elapsed_ms": elapsed * 1000,
            "success": True
        }
    
    def run_test_suite(self, ground_truth_file: str) -> Dict:
        """Run full test suite and return summary."""
        print(f"\nLoading test cases from {ground_truth_file}...")
        test_cases = self.load_ground_truth(ground_truth_file)
        
        if not test_cases:
            print("No test cases found!")
            return {}
        
        print(f"Found {len(test_cases)} test cases\n")
        print("="*80)
        print("RUNNING VERIFICATION TESTS")
        print("="*80)
        
        results = []
        correct_count = 0
        verified_count = 0
        total_time = 0
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {case['image_path']}")
            print(f"  Expected: {case['true_id']} - {case['true_name']}")
            
            result = self.test_single_card(
                case['image_path'],
                case['true_id'],
                case['true_name']
            )
            
            if result['success']:
                results.append(result)
                total_time += result['elapsed_ms']
                
                if result['correct']:
                    correct_count += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ INCORRECT"
                
                if result['verified']:
                    verified_count += 1
                    verify_status = "(auto-verified)"
                else:
                    verify_status = "(needs review)"
                
                print(f"  {status} {verify_status}")
                print(f"  Predicted: {result['predicted_id']} - {result['predicted_name']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Scores: FAISS={result['scores'].get('faiss', 0):.3f}, "
                      f"Template={result['scores'].get('template', 0):.3f}, "
                      f"Hash={result['scores'].get('hash', 0):.3f}, "
                      f"SSIM={result['scores'].get('ssim', 0):.3f}")
                print(f"  Time: {result['elapsed_ms']:.1f}ms")
            else:
                print(f"  ✗ ERROR: {result.get('error', 'Unknown error')}")
        
        # Calculate metrics
        total = len(results)
        accuracy = correct_count / total if total > 0 else 0
        verification_rate = verified_count / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total test cases: {total}")
        print(f"Correct predictions: {correct_count} ({accuracy:.1%})")
        print(f"Auto-verified: {verified_count} ({verification_rate:.1%})")
        print(f"Needs manual review: {total - verified_count} ({1-verification_rate:.1%})")
        print(f"Average time per card: {avg_time:.1f}ms")
        print("="*80)
        
        # Breakdown by difficulty (if available)
        difficulties = {}
        for case, result in zip(test_cases, results):
            diff = case.get('difficulty', 'unknown')
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'correct': 0}
            difficulties[diff]['total'] += 1
            if result['correct']:
                difficulties[diff]['correct'] += 1
        
        if difficulties:
            print("\nAccuracy by Difficulty:")
            for diff, stats in sorted(difficulties.items()):
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {diff.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.1%})")
        
        # Find failure cases
        failures = [r for r in results if not r['correct']]
        if failures:
            print(f"\nFailure Cases ({len(failures)}):")
            for fail in failures:
                print(f"  - {fail['image_path']}")
                print(f"    Expected: {fail['true_id']}, Got: {fail['predicted_id']}")
                print(f"    Confidence: {fail['confidence']:.2%}")
        
        summary = {
            'total': total,
            'correct': correct_count,
            'accuracy': accuracy,
            'verified': verified_count,
            'verification_rate': verification_rate,
            'avg_time_ms': avg_time,
            'results': results,
            'failures': failures
        }
        
        self.test_results = results
        return summary
    
    def export_results(self, output_file: str = "test_results.json"):
        """Export test results to JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nTest results exported to: {output_file}")
    
    def create_failure_report(self, output_file: str = "failure_report.html"):
        """Create HTML report showing failure cases with images."""
        failures = [r for r in self.test_results if not r['correct']]
        
        if not failures:
            print("No failures to report!")
            return
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Verification Failure Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .failure { border: 1px solid #ccc; padding: 15px; margin: 15px 0; }
        .failure h3 { color: #d00; }
        .scores { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .score-bar { background: #ddd; height: 20px; margin: 5px 0; }
        .score-fill { background: #4CAF50; height: 100%; }
        .score-fill.low { background: #f44336; }
        .score-fill.medium { background: #ff9800; }
    </style>
</head>
<body>
    <h1>Verification Failure Report</h1>
    <p>Total failures: """ + str(len(failures)) + """</p>
"""
        
        for i, fail in enumerate(failures, 1):
            scores = fail.get('scores', {})
            
            html += f"""
    <div class="failure">
        <h3>Failure #{i}: {fail['image_path']}</h3>
        <p><strong>Expected:</strong> {fail['true_id']} - {fail['true_name']}</p>
        <p><strong>Predicted:</strong> {fail['predicted_id']} - {fail['predicted_name']}</p>
        <p><strong>Overall Confidence:</strong> {fail['confidence']:.2%}</p>
        
        <div class="scores">
            <h4>Verification Scores:</h4>
"""
            
            for score_name, score_val in scores.items():
                bar_class = "low" if score_val < 0.7 else ("medium" if score_val < 0.85 else "")
                html += f"""
            <div>
                <strong>{score_name.upper()}:</strong> {score_val:.3f}
                <div class="score-bar">
                    <div class="score-fill {bar_class}" style="width: {score_val*100}%"></div>
                </div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Failure report created: {output_file}")

def create_sample_ground_truth():
    """Create a sample ground truth file for testing."""
    sample_data = [
        {
            "image_path": "test_identifier/card_01.png",
            "true_id": "89631139",
            "true_name": "Blue-Eyes White Dragon",
            "difficulty": "easy"
        },
        {
            "image_path": "test_identifier/card_02.png",
            "true_id": "46986414",
            "true_name": "Dark Magician",
            "difficulty": "easy"
        },
        {
            "image_path": "test_identifier/card_03.png",
            "true_id": "55144522",
            "true_name": "Pot of Greed",
            "difficulty": "medium"
        },
        {
            "image_path": "test_identifier/card_04.png",
            "true_id": "12345678",
            "true_name": "Unknown Card",
            "difficulty": "hard"
        }
    ]
    
    output_file = "ground_truth_sample.json"
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample ground truth created: {output_file}")
    print("Edit this file with your actual card IDs and names")
    return output_file

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test verification system accuracy")
    parser.add_argument('--ground-truth', '-g', type=str, 
                       help='Path to ground truth JSON file')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample ground truth file')
    parser.add_argument('--export', '-e', type=str, default='test_results.json',
                       help='Export results to JSON file')
    parser.add_argument('--report', '-r', type=str, default='failure_report.html',
                       help='Create HTML failure report')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_ground_truth()
        return
    
    if not args.ground_truth:
        print("Error: Please provide --ground-truth file or use --create-sample")
        print("\nUsage:")
        print("  python test_verification_system.py --create-sample")
        print("  python test_verification_system.py --ground-truth ground_truth.json")
        return
    
    # Run tests
    tester = VerificationTester()
    summary = tester.run_test_suite(args.ground_truth)
    
    if summary:
        # Export results
        tester.export_results(args.export)
        
        # Create failure report if there are failures
        if summary['failures']:
            tester.create_failure_report(args.report)
        
        # Final verdict
        print("\n" + "="*80)
        if summary['accuracy'] >= 0.99:
            print("✓ EXCELLENT: Accuracy >= 99%")
        elif summary['accuracy'] >= 0.95:
            print("✓ GOOD: Accuracy >= 95%")
        elif summary['accuracy'] >= 0.90:
            print("⚠ ACCEPTABLE: Accuracy >= 90% (consider improvements)")
        else:
            print("✗ POOR: Accuracy < 90% (needs improvement)")
        print("="*80)

if __name__ == "__main__":
    main()
