# Saved prediction scoring

`gsm8k_scoring.py` and `evaluate_saved.py` require only Python 3.10+ standard libraries. They do not import model libraries, access the network, deserialize weights, or generate predictions. `numeric-v2` corrects numeric formatting issues in the historical final notebook; it is a new scoring protocol, not a reproduction of the printed historical accuracies.

## Evidence currently available

- `eval-gsm8k.ipynb` retains historical printed final-test results: base 41.17%, SFT 44.50%, RL 52.46%. The inspected notebook evaluated all 1,319 test rows using temperature 0 and a 512-token completion limit, with different system prompts for RL versus base/SFT. Exact generated final-test completions are not stored alongside it.
- Three local `eval_*_500.csv` files retain internal-evaluation outputs. They are ignored by the existing Git configuration. These use different historical parsers and cannot support the final-test result without the final-test completions. RL rows include prompt echo, whereas the new evaluator requires completion-only text. Do not strip prompts heuristically and call the result an unchanged historical experiment.
- The internal-evaluation decline cannot be attributed exclusively to the parser from those numbers alone. Formatting, decoding, prompt echo, and model behavior are confounded. A controlled comparison requires identical questions, decoding, completion scope, and a documented extraction policy, with prompts/model adapters recorded.
- Adapter ZIPs are present but were not loaded, unzipped, hashed, or executed during this work. GPU times and notebook metrics are historical artifacts, not newly measured compute costs.

## Input contract

Use one JSON object per line with nonempty unique `id`, `prediction` (completion text or null for failed/missing generation), and `gold` (reference answer text). Store every expected example; a null prediction stays in the accuracy denominator. Do not include input prompts in `prediction`.

Supply a separate run metadata JSON with:

| Field | Required meaning |
|---|---|
| `dataset`, `dataset_revision`, `split` | Dataset identity, immutable dataset revision, exact split name/selection |
| `split_sha256` | SHA-256 of the ordered ID/gold manifest as described below |
| `model`, `model_revision` | Model ID and immutable checkpoint revision or weights digest |
| `adapter_revision` | Adapter weights revision/digest, or null for the base model |
| `decoding` | Object containing numeric `temperature`, positive integer `max_new_tokens`, and integer `seed` (null when unrecorded); include other actual generation settings |
| `prompt_sha256` | SHA-256 of the exact UTF-8 prompt/template file; preserve that file alongside the run |
| `completion_scope` | Must be `"completion_only"` |
| `expected_count` | Positive expected example count before generation |

Compute `split_sha256` from an ordered list of `{"id": ..., "gold": ...}` objects corresponding to the JSONL order, serialized using `json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")`. This checks the ID/reference manifest, not dataset authenticity or question content. Preserve a separate authoritative dataset manifest containing questions and source row IDs. Do not invent revisions or replace unknown settings with guessed values. Missing provenance must be recovered before labeling a report reproducible.

```powershell
python evaluate_saved.py --predictions path/to/completions.jsonl --metadata path/to/run.json --output path/to/new-report.json
python -m unittest test_offline_scoring.py
```

The output path must be new. Reports bind exact input bytes, run metadata, scoring module, and CLI through SHA-256 hashes; retain all files together. They include every example's extracted values/errors and extraction method. No historical CSVs were rescored in this change. Synthetic unit-test inputs are solely software checks and are not portfolio evidence.

## Extraction and correctness

1. Use the last complete `<answer>...</answer>` block, ignoring case.
2. Otherwise use the suffix after the final `####`.
3. Otherwise use the last numeric token from completion text.

The last numeric token within the selected segment is parsed exactly using rational arithmetic. Signed integers, decimals, valid thousands separators, simple fractions, and scientific notation are supported. Percent tokens retain their written numeric value; no unit conversion is performed. Mixed fractions and malformed numeric grouping fail explicitly. Missing or malformed explicit answer blocks do not borrow an answer from earlier reasoning. Excessively large numeric strings/exponents fail closed.

Correctness uses strict absolute error `< 0.0001`, preserving the final notebook's tolerance rule without binary floating-point rounding. The CLI records an optional `--tolerance` override. Invalid gold references stop report creation. Unparseable predictions count as incorrect.

This parser is deliberately lenient about surrounding answer prose and cannot infer semantic intent from arbitrary text. Numeric-fallback answers should be audited separately, and format compliance should be reported separately from answer correctness.

## Notebook integration

The final evaluation notebook imports the shared scorer for future executions and checks prediction/gold/prompt counts before scoring. Only two source cells were changed. Stored historical outputs remain unchanged and predate v2; they were not rerun. Training rewards and historical internal-evaluation notebook functions remain unchanged so their original experimental meaning is preserved. Future generation should save completion-only per-example JSONL and complete provenance before claiming a reproducible comparison.
