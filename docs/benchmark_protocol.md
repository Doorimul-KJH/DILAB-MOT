# Benchmark Protocol

## Quantitative Evaluation

Quantitative evaluation should use MOTChallenge or another dataset with ground truth annotations. Results from unannotated videos are qualitative only.

TrackEval will be used for benchmark metrics.

## Metrics

The benchmark should track:

- HOTA
- MOTA
- IDF1
- ID Switches

## Fair Comparison

When comparing trackers, detection input must be fixed and cached. A tracker comparison is not fair if each tracker receives different detector outputs unless the experiment is explicitly defined as a full pipeline comparison.

Experiment reports should record the dataset split, detection source, tracker preset, configuration file, code version, environment information, and output paths.

## Qualitative Inputs

Arbitrary YouTube videos and other unannotated videos are qualitative demo inputs unless annotated ground truth exists. They can be used for visual inspection, robustness checks, and failure case discussion, but not as quantitative benchmark evidence.
