# Reference-relative JEV analysis

JEV sees a question, model-generated/translated reference answer, and candidate.
It does not see the constitutional article. Scores measure reference equivalence,
not independently established legal correctness or calibrated Urdu probabilities.

## Archived run

The root `data/automated_evaluations_qa/jev` files retain the original 1,116
judgments, returned model `jev-1.13.0`, on nine models' answers to 124 questions.
The archived prompt incorrectly calls references human-expert-written. These
results are explicitly reported under that original prompt, with copied human
labels refreshed after logged corrections. The mT5 run uses its annotated variant.

```sh
make jev-agreement
.venv/bin/python -m scripts.jev.threshold_sweep --no-plot
```

These commands are offline and do not require typesafe-sdk or API keys.
Install matplotlib only to regenerate the optional plot. Agreement reports
confusion counts, positive recall, specificity, and the always-positive baseline.
Kappa uses the same definition in agreement and threshold analysis, including
constant predictions (undefined only when chance agreement is one).

Threshold selection on these same labels is exploratory tuning. Split by question
or article for future calibration; do not split a question's different model
answers across folds. Individual judgments are not independent questions.

## Revised prompt for a new inference run

`pipeline.py` now accurately states reference provenance. New runs default to
`data/automated_evaluations_qa/jev/reference-v2/`, with prompt/input fingerprints.
Legacy checkpoints are never reused under a changed prompt. An unversioned or
changed-input checkpoint raises an error rather than silently mixing results.

```sh
make extras                 # includes typesafe-sdk; inference-only dependency
make env                    # add TYPESAFE_API_KEY or JEV_API_KEY locally
make jev-run MODEL=claude-sonnet-5
```

New inference makes provider calls. No revised-prompt judgments are included
with the saved data. Retain legacy files; use a new output
directory when changing inputs or criteria. Rerunning moving judge aliases does
not reproduce a historical snapshot. A notebook should `await run_model(...)`
inside its existing event loop rather than call the synchronous wrapper.
