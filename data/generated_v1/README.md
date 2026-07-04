# LWY-bench Generated V1

This is the first synthetic large-scale draft of LWY-bench.

## Scale

- Base cases: 336
- Domains: 7
- Cases per domain: 48
- Single-turn instances: 336
- Multi-turn dialogues: 336
- Turns per multi-turn dialogue: 5
- Total model-response opportunities: 2016

## Directory Contract

Each base case has its own folder:

```text
cases/case_XXXX_<domain>_<topic>/
├── base.json
├── misleading_context.md
├── single_turn/
│   └── input.json
└── multi_turn/
    └── dialogue.json
```

## Domains

| Domain | Cases |
|---|---:|
| pseudo_math_proof | 48 |
| pseudo_science_mechanism | 48 |
| fake_empirical_study | 48 |
| medical_misinformation | 48 |
| legal_policy_misreading | 48 |
| economic_causal_claim | 48 |
| computer_science_theory | 48 |

## Indexes

```text
indexes/cases.jsonl
indexes/single_turn.jsonl
indexes/multi_turn.jsonl
manifests/dataset_card.json
```

## Important Caveat

This is a synthetic draft dataset. It is suitable for early pipeline testing and model probing, but it requires human audit before being treated as a final public benchmark.
