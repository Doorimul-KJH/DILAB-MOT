# Project Principles

## Paper-Level Selection

DILAB-MOT is a paper-based MOT reproduction and benchmark platform. The primary user action is selecting a paper or algorithm preset, not composing arbitrary detectors and trackers.

Paper presets define the complete pipeline needed to reproduce or benchmark that paper's method. Detector input, motion model, association method, output format, and evaluation protocol are part of the preset.

## SORT Paper Mode Behavior

SORT paper mode must follow the SORT paper structure:

- Faster R-CNN detections or MOTChallenge public detections
- Kalman Filter motion model
- IoU matching
- Hungarian Algorithm association
- SORT tracking output

Selecting SORT should execute the SORT paper pipeline.

## No Silent YOLO Substitution

YOLO must not be silently substituted into SORT paper reproduction mode. If YOLO support is added later, it must live in an explicitly separate demo or playground mode, not as the default SORT paper pipeline.

## Shared CLI and GUI Runner

CLI execution and future GUI execution must call the same core runner logic, such as PaperPipeline or ExperimentRunner. The GUI must be a frontend wrapper and must not own tracking behavior.

## Benchmark and Demo Separation

Quantitative benchmark workflows and qualitative demo workflows must remain separate.

Quantitative benchmarks require fixed protocols, comparable inputs, recorded configuration, and datasets with ground truth. Qualitative demos can show robustness or visual behavior but should not be reported as benchmark evidence unless they include valid annotations.

## YouTube and Arbitrary Videos

YouTube videos and arbitrary user videos are demo or robustness-check inputs by default. They are not quantitative benchmark inputs unless ground truth annotations exist and the evaluation protocol is documented.
