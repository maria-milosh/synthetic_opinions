# Synthetic Opinions & Cross-Pollination Experiment

This repository contains code to generate **synthetic opinions using LLMs**, represent them as **embeddings**, and analyze their **similarity and diversity** as a foundation for cross-pollination experiments in collective decision-making.

The core use case is to study how exposure to others’ arguments (cross-pollination) affects reasoning, confidence, and choices under limited information.

---

## Project structure
```
.
├── personas.jsonl # Synthetic participant personas
├── prompt_template.txt # Prompt template with placeholders
├── call_llm.py # Generate synthetic responses via LLM
├── get_embeddings.py # Create embeddings (local)
├── outputs/run_*
│ ├── transcripts.jsonl # Raw + parsed LLM responses
│ ├── embeddings_*.jsonl # Embeddings for different text fields
│ └── similarity_plots/ # Distance plots
└── README.md
```

---

## Personas

`personas.jsonl` defines participant types with controlled variation.

- Personas preserve **core types** (e.g. self-optimizer, egalitarian, meritocratic, civic)
- Multiple variants introduce noise in background and phrasing
- Personas are *narrative only* — no explicit utility functions are exposed

Each line is a standalone JSON object.

---

## Generating synthetic responses

`call_llm.py`:
- Loads personas
- Renders prompts using `prompt_template.txt`
- Queries the LLM
- Stores both raw text and parsed JSON for auditability, and query details

Outputs are written to: outputs/transcripts.jsonl

Each record includes:
- persona ID
- full prompt text
- raw response text
- parsed JSON
- timestamps and metadata

---

## Creating embeddings

`get_embeddings.py` supports fully local embeddings (SentenceTransformers)

You can embed different fields independently:
- `argument_snippet`
- `justification_bullets`
- other parsed response fields

Minimal preprocessing is applied:
- whitespace normalization only
- no stopword removal, stemming, or summarization

This preserves **normative stance and reasoning structure**.

___

## Distance analysis

From embeddings, the analysis code computes:
+ cosine similarity and distance matrices
+ pairwise distance distributions
+ within-persona vs between-persona distances
+ persona-pair distance comparisons
+ heatmaps and histograms with the above

These distances can be used to:
+ validate that embeddings capture stance
+ design cross-pollination exposure rules
+ define “ideological distance” quantitatively

___

Set API key: `export OPENAI_API_KEY="your_key_here"`