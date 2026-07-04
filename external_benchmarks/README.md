# External Benchmark Sources

This folder stores public sycophancy benchmark resources related to the papers in `03_Sycophancy`.

| Source | Status | Normalized data | Notes |
|---|---|---:|---|
| ChaosWithKeywords | downloaded_and_normalized | 2500 | The repo provides misleading keyword prompts and source model statements. It does not include verified gold factuality labels in the CSV files. |
| sycophancy-eval | downloaded_and_normalized | 20656 | Repository contains prompt datasets but no explicit license file. Use with citation and verify license before public redistribution. |
| sycophancy-intervention | repo_downloaded_and_simple_addition_eval_generated | 5000 | The repository provides code to generate training/eval data. This normalized file deterministically instantiates the simple addition false-claim evaluation without requiring HuggingFace datasets. |
| SycEval | no_public_repo_found_in_pdf_or_web_search | 0 | Paper describes AMPS and MedQuad based evaluation, but no directly downloadable benchmark repository was found in the local PDF text or web search. |
| Check My Work | no_public_repo_found_in_pdf_or_web_search | 0 | Paper describes MMLU-based educational framing experiments, but no directly downloadable benchmark repository was found in the local PDF text or web search. |

Raw repositories are stored under `raw_repos/`; normalized JSONL files are stored under `normalized/`.
