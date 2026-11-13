"""Small synthetic test cases only; no historical accuracy recomputation."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate_saved import build_report, sha256
from gsm8k_scoring import extract_answer, score_answer


def fixture_bytes():
    rows = [{'id': 'unit-1', 'prediction': '<answer>1/2</answer>', 'gold': '#### 0.5'},
            {'id': 'unit-2', 'prediction': None, 'gold': '#### -2'}]
    split = [{'id': row['id'], 'gold': row['gold']} for row in rows]
    metadata = {'dataset': 'synthetic-unit-test', 'dataset_revision': 'fixture-v1', 'split': 'unit-test',
                'split_sha256': sha256(json.dumps(split, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()),
                'model': 'no-model-test-fixture', 'model_revision': 'no-model-executed', 'adapter_revision': None,
                'decoding': {'temperature': 0, 'max_new_tokens': 1, 'seed': None},
                'prompt_sha256': sha256(b'no prompt; synthetic unit test'), 'completion_scope': 'completion_only', 'expected_count': 2}
    return '\n'.join(json.dumps(row) for row in rows).encode(), json.dumps(metadata).encode()


class ExtractionTests(unittest.TestCase):
    def test_priority_formats(self):
        extracted = extract_answer('Reasoning 9. #### 7 <answer>5</answer> trailing 100')
        self.assertEqual((extracted.value, extracted.source), (5, 'xml'))
        self.assertEqual(extract_answer('Earlier 4. #### 6').value, 6)
        self.assertEqual(extract_answer('The result is -12 dollars.').value, -12)

    def test_last_answer_block(self):
        self.assertEqual(extract_answer('<answer>1</answer><answer>2</answer>').value, 2)

    def test_numeric_forms(self):
        for text, value in [('-1,200.50', Fraction('-1200.5')), ('−.5', Fraction('-0.5')),
                            ('+2.5', Fraction('2.5')), ('3/4', Fraction(3, 4)),
                            ('-3 / 2', Fraction(-3, 2)), ('1.2e3', Fraction(1200))]:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text).value, value)

    def test_invalid_values(self):
        for text in ['', 'unknown', 'NaN', 'Infinity', '1,23', '1/0', '2 1/2', '<answer>4', '1e9999']:
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text).value)

    def test_explicit_invalid_answer_does_not_borrow_reasoning(self):
        for text in ['Reasoning 42. <answer>unknown</answer>', 'Reasoning 42. #### unknown']:
            self.assertIsNone(extract_answer(text).value)

    def test_numeric_equality_and_strict_tolerance(self):
        self.assertTrue(score_answer('<answer>1,000</answer>', '#### 1000')['correct'])
        self.assertTrue(score_answer('1/2', '0.5')['correct'])
        self.assertFalse(score_answer('1.0001', '1')['correct'])
        self.assertTrue(score_answer('1.00009', '1')['correct'])

    def test_missing_predictions_fail_and_bad_gold_errors(self):
        self.assertFalse(score_answer(None, '#### 1')['correct'])
        with self.assertRaises(ValueError):
            score_answer('0', 'missing gold')


class OfflineReportTests(unittest.TestCase):
    def test_report_retains_missing_predictions_and_hashes(self):
        predictions, metadata = fixture_bytes()
        report = build_report(predictions, metadata)
        self.assertEqual(report['count'], 2)
        self.assertEqual(report['correct'], 1)
        self.assertEqual(report['unparseable_predictions'], 1)
        self.assertEqual(report['hashes']['predictions_sha256'], sha256(predictions))

    def test_metadata_and_coverage_fail_closed(self):
        predictions, metadata = fixture_bytes()
        for mutate in [lambda m: m.pop('model_revision'), lambda m: m.update(expected_count=3),
                       lambda m: m.update(completion_scope='prompt_and_completion'),
                       lambda m: m.update(split_sha256='0' * 64)]:
            value = json.loads(metadata)
            mutate(value)
            with self.assertRaises(ValueError):
                build_report(predictions, json.dumps(value).encode())

    def test_duplicate_ids_rejected(self):
        predictions, metadata = fixture_bytes()
        first = predictions.splitlines()[0]
        with self.assertRaises(ValueError):
            build_report(first + b'\n' + first, metadata)

    def test_cli_writes_once_and_preserves_input(self):
        predictions, metadata = fixture_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'predictions.jsonl').write_bytes(predictions)
            (root / 'metadata.json').write_bytes(metadata)
            args = [sys.executable, str(Path(__file__).with_name('evaluate_saved.py')),
                    '--predictions', str(root / 'predictions.jsonl'), '--metadata', str(root / 'metadata.json'),
                    '--output', str(root / 'report.json')]
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            report_bytes = (root / 'report.json').read_bytes()
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)
            self.assertEqual((root / 'report.json').read_bytes(), report_bytes)
            self.assertEqual((root / 'predictions.jsonl').read_bytes(), predictions)


if __name__ == '__main__':
    unittest.main()
