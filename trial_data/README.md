# DiCo-NLI English Trial Data Bundle

The trial data folder contains the English trial data for SemEval 2027 Task 2
DiCo-NLI. The Spanish, Basque, and mixed-language trial files, multilingual
translation audits, and organizer JSON have been removed from this local
workspace because only the English track will be used.

## Track Files

The English bundle contains:

- `*_participant_labeled.csv`: trial/train-style participant file with labels.
- `*_participant_unlabeled.csv`: test-style participant file without labels.
- `*_hidden_gold.csv`: fixed hidden gold file consumed by the scorer.
- `*_submission_template.csv`: minimal `instance_id,label` submission template.

## Baselines

`baselines/random/` and `baselines/majority/` contain English-track naive
participant submission files and scorer outputs. `baselines/baseline_summary.json`
summarizes weighted F1, SoftCons, and HardCons for the English track.
