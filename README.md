# Playing with GRPO: Teaching Mistral-7B to “Think”

I wanted to experiment with **GRPO (Group Relative Policy Optimization)** to see whether I could induce _explicit reasoning behavior_, similar in spirit to DeepSeek-R1, on a smaller open model.

The setup is intentionally simple:

- Start with **Mistral-7B Instruct v0.3**
- Warm it up with **SFT on GSM8K**
- Apply **GRPO** to encourage structured reasoning via XML-style `<reasoning>` traces
- Measure whether this _actually improves math accuracy_, not just verbosity

---

## Experiment Setup

### Model & Dataset

- **Model:** [unsloth/mistral-7b-instruct-v0.3](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3)
- **Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k)

To keep evaluation clean and avoid leakage, the GSM8K _training_ split was subdivided as follows:

| Split Source | Usage         | Count |
| ------------ | ------------- | ----- |
| Train        | SFT           | 4,000 |
| Train        | RL (GRPO)     | 2,900 |
| Train        | Internal Eval | 500   |
| Test         | Final Eval    | 1,319 |

---

### Hardware

Different stages had very different memory profiles:

- **Zero-shot baseline & SFT**
  Kaggle - _2× T4 (15GB VRAM each)_

- **RL (GRPO) & Final Evaluation**
  LightningAI Studio - _1× L4 (24GB VRAM)_
  vLLM was used during evaluation for faster generation.

---

## Results

These are historical notebook outputs. They were not rerun during the offline evaluation work, and final-test per-example completions are not present locally. [The offline scoring protocol](OFFLINE-EVALUATION.md) documents a new shared scorer and the evidence needed for a reproducible comparison.

### 1. GSM8K Test Set (Unseen, Final Numbers)

The final evaluation notebook records the following results on the 1,319-row test split.

| Stage         | Accuracy   | Notes                       |
| ------------- | ---------- | --------------------------- |
| Zero-Shot     | 41.17%     | Mistral-7B Instruct v0.3    |
| SFT           | 44.50%     | 4k supervised examples      |
| **RL (GRPO)** | **52.46%** | Structured reasoning via RL |

The recorded difference is **7.96 percentage points** over SFT. This is a historical result under the notebook's prompts, decoding, and extraction rules, not a newly reproduced gain.

---

### 2. Internal Train-Eval Split (500 examples)

| Stage     | Accuracy |
| --------- | -------- |
| Zero-Shot | 41.20%   |
| SFT       | 58.00%   |
| RL        | 45.40%\* |

- Different evaluation paths make this an uncontrolled comparison. The parser alone does not establish whether model quality regressed.

---

## Why Does RL Look Worse on the Train-Eval Split?

At first glance, RL appears to _hurt_ performance on the 500-example internal eval split (58.00% → 45.40%).
This initially raised concerns.

The notebooks used different extraction and comparison functions. SFT internal evaluation compared the last extracted numeric string; RL internal evaluation used numeric comparison, removed commas, and decoded the input prompt together with the completion. The code does not require well-formed XML to award correctness when tags are missing, so missing tags alone do not explain the discrepancy.

The final evaluation notebook used a common extraction function for all three models, preferring XML answer tags, then `####`, then the last numeric token. Its regex still mishandles thousands separators and fractions, and RL used a different system prompt from the other stages. Equal parsing helps comparability but does not isolate the effect of training.

### Shared offline scorer

A standard-library scorer now handles signed values, decimals, valid comma grouping, and simple fractions, with one recorded tolerance rule. The CLI scores saved completion-only JSONL, retains failures in the denominator, checks the ordered reference manifest, and writes a report with input/protocol hashes and required model/split/decoding metadata.

See [OFFLINE-EVALUATION.md](OFFLINE-EVALUATION.md) for usage, numeric semantics, artifact limitations, and the tests. Future final-notebook executions use this shared scorer; the stored historical outputs predate it. No historical accuracy was recomputed for this change.

---

## Experiment Flow

The project follows a standard alignment pipeline:

1. **`sft-gsm8k.ipynb` - Supervised Fine-Tuning**

   - Teaches the base model GSM8K-style answering
   - Establishes a strong baseline

2. **`rl-gsm8k.ipynb` - Reinforcement Learning (GRPO)**

   - Encourages explicit reasoning traces using XML tags
   - Rewards correctness of the final numeric answer
   - Trades some formatting stability for better reasoning

3. **`eval-gsm8k.ipynb` - Final Evaluation**

   - Uses vLLM for fast generation
   - Applies a unified, format-agnostic correctness function

---

## What this experiment establishes

The stored notebooks document an SFT-to-GRPO experiment and report a higher final-test accuracy after GRPO. The strongest engineering lesson is the need to version correctness rules, preserve completion-only predictions, and separate format compliance from mathematical correctness. A controlled rerun or complete saved final-test predictions are still needed to verify the historical comparison under the new scoring protocol.
