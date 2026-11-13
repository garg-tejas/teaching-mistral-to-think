"""Score immutable saved JSONL completions; never runs or downloads a model."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from fractions import Fraction

from gsm8k_scoring import PROTOCOL_VERSION, score_answer


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_metadata(metadata):
    required = ['dataset', 'dataset_revision', 'split', 'split_sha256', 'model', 'model_revision',
                'adapter_revision', 'decoding', 'prompt_sha256', 'completion_scope', 'expected_count']
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f'Missing provenance fields: {", ".join(missing)}')
    for key in ['dataset', 'dataset_revision', 'split', 'model', 'model_revision']:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f'{key} must identify the actual run')
    for key in ['split_sha256', 'prompt_sha256']:
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{key} must be a lowercase SHA-256 digest')
    if metadata['completion_scope'] != 'completion_only':
        raise ValueError('Prediction text must exclude the input prompt')
    if type(metadata['expected_count']) is not int or metadata['expected_count'] < 1:
        raise ValueError('expected_count must be a positive integer')
    if not isinstance(metadata['decoding'], dict) or not all(k in metadata['decoding'] for k in ['temperature', 'max_new_tokens', 'seed']):
        raise ValueError('decoding requires temperature, max_new_tokens, and seed (null if unrecorded)')
    decoding = metadata['decoding']
    if type(decoding['temperature']) not in (int, float) or not 0 <= decoding['temperature'] < float('inf'):
        raise ValueError('temperature must be finite and nonnegative')
    if type(decoding['max_new_tokens']) is not int or decoding['max_new_tokens'] < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if decoding['seed'] is not None and type(decoding['seed']) is not int:
        raise ValueError('seed must be an integer or null')
    if metadata['adapter_revision'] is not None and not isinstance(metadata['adapter_revision'], str):
        raise ValueError('adapter_revision must identify adapter weights or be null for base model')
    return metadata


def build_report(prediction_bytes, metadata_bytes, tolerance='0.0001'):
    metadata = validate_metadata(json.loads(metadata_bytes))
    rows, split_rows, ids = [], [], set()
    for number, line in enumerate(prediction_bytes.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(key in row for key in ['id', 'prediction', 'gold']):
            raise ValueError(f'Line {number}: requires id, prediction, and gold')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError(f'Line {number}: id must be nonempty and unique')
        if row['prediction'] is not None and not isinstance(row['prediction'], str):
            raise ValueError(f'Line {number}: prediction must be text or null')
        if not isinstance(row['gold'], str):
            raise ValueError(f'Line {number}: gold must be text')
        ids.add(row['id'])
        split_rows.append({'id': row['id'], 'gold': row['gold']})
        rows.append({'id': row['id'], **score_answer(row['prediction'], row['gold'], tolerance)})
    if len(rows) != metadata['expected_count']:
        raise ValueError('Row count differs from expected_count; retain failed generations as null predictions')
    split_hash = sha256(json.dumps(split_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    if split_hash != metadata['split_sha256']:
        raise ValueError('Ordered id/gold manifest differs from split_sha256')
    correct = sum(row['correct'] for row in rows)
    return {
        'protocol': PROTOCOL_VERSION,
        'comparison': 'strict_absolute_difference_less_than_tolerance',
        'tolerance': str(Fraction(tolerance)),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': metadata,
        'hashes': {'predictions_sha256': sha256(prediction_bytes), 'metadata_sha256': sha256(metadata_bytes),
                   'scorer_sha256': sha256(Path(__file__).with_name('gsm8k_scoring.py').read_bytes()),
                   'cli_sha256': sha256(Path(__file__).read_bytes())},
        'count': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
        'unparseable_predictions': sum(row['prediction_value'] is None for row in rows),
        'extraction_sources': dict(Counter(row['extraction_source'] for row in rows)),
        'per_example': rows,
        'limitations': ['Offline rescoring only; not new inference or a reproduction of historical notebook accuracy.',
                        'Run metadata is caller supplied; hashes bind files, not the truth of provenance claims.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tolerance', default='0.0001')
    args = parser.parse_args()
    try:
        report = build_report(args.predictions.read_bytes(), args.metadata.read_bytes(), args.tolerance)
        # Exclusive creation protects historical reports and inputs from replacement.
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(f"Saved offline report: {args.output} ({report['count']} examples)")


if __name__ == '__main__':
    main()
