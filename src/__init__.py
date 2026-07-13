"""
Reference implementation for "Does Privacy Distort Alignment? Decomposing the
Effects of Differential Privacy on LLM Hallucination and Misalignment".

Fine-tunes a real small open-weight Hugging Face model under 4 privacy
conditions across an epsilon grid, evaluating truthfulness/hallucination
(TruthfulQA), instruction-following (IFEval), helpfulness (AlpacaDataCleaned),
and safety/refusal (a hand-written implementation), then runs a
mediation/decomposition analysis.
"""
