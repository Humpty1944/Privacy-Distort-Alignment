# Privacy-Distort-Alignment
Decomposing the Effects of Differential Privacy on LLM Hallucination and Misalignment

# Center Research Questions:
When DP degrades LLM alignment, is the degradation primarily caused by measurable semantic distortion of the training/inference signal, or by other effects of DP noise such as reduced optimization quality, calibration shifts, or altered refusal/helpfulness trade-offs? Whether distorted meaning can cause errors;

Whether semantic distortion explains most DP-induced alignment degradation after controlling for alternative mechanisms.

# Scope
This project studies how semantic differentially private mechanisms affect LLM alignment. It focuses on a feasible, controlled setting: small open-weight language models, public alignment datasets, and lightweight fine-tuning or inference-time differentially private perturbation methods. 

The project tests the conjecture that the relationship between privacy strength and alignment quality is not simply monotonic. DP may harm, preserve, or occasionally improve alignment depending on whether privacy noise changes task-relevant semantics and how alignment regularization responds.

# Importance
It separates the privacy-utility trade-off into more precise components. Rather than asking whether privacy “adds noise”, it asks which kinds of noise actually damage alignment and which can be controlled through mechanism design or alignment regularization.

# Goals
Understand how DP mechanisms affect LLM utility and safety, then separate observed effects into semantic and non-semantic pathways.
* **Direct privacy effect**: changes in training dynamics, generation behavior, or calibration caused by the privacy mechanism itself.
* **Semantic distortion pathway**: privacy noise changes task-relevant meaning such as intent, entities, facts, or relations.
* **Other aligment pahtway**: optimization instability, uncertainty miscalibration, refusal/helpfulness shifts, or regularization effects.

# Expected Deliverables
1. Empirical evidence on how DP mechanism affects LLMs in utility and safety that may lead to a clear conjecture.
2. Decomposition of DP effects into semantic distortion and non-semantic pathways.
3. A compact evaluation suite combining truthfulness, hallucination, safety, and instruction-following tasks at different privacy budgets and semantic-distortion levels.
4. A small benchmark set of open-weight LLMs under several baseline privacy mechanisms: no privacy; random token replacement or deletion; semantic text sanitization inspired by SanText-style local DP; DP fine-tuning using DP-SGD or DP-compatible parameter-efficient tuning.
5. Mitigation strategies that preserve task-relevant semantics under DP guarantees.
7. A reproducible codebase and short paper.

# Plan
* **Wk 1**: Setup, literature map, metrics, benchmark.
* **Wk 2**: DP mechanisms and pilot experiments.
* **Wk 3**: privacy–-semantic distortion-alignment curves across at least two datasets.
* **Wk 4**: mitigation ideas, paper, and code.
