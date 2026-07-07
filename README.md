# DILAB-MOT

DILAB-MOT is a paper-based multi-object tracking reproduction and benchmark platform for the DILAB undergraduate researcher project.

The project is organized around papers and algorithms, not arbitrary detector/tracker combinations. Users select a paper pipeline such as SORT, DeepSORT, StrongSORT, OC-SORT, or ByteTrack. The selected paper determines the detector input, tracking logic, association method, evaluation protocol, and outputs.

## Core Principle

Users select papers, not free-form detector and tracker combinations.

Correct paper-level selections:

- SORT
- DeepSORT
- StrongSORT
- OC-SORT
- ByteTrack

Incorrect default concept:

- Detector: YOLO / Faster R-CNN / etc.
- Tracker: SORT / DeepSORT / etc.

SORT paper mode must not use YOLO. Selecting SORT should execute the SORT paper pipeline: Faster R-CNN detections or MOTChallenge public detections, Kalman Filter motion model, IoU matching, Hungarian Algorithm association, and SORT tracking output.

YOLO may be introduced later only as a separate demo or playground mode. It must not be silently substituted into SORT paper reproduction mode.

## Planned Papers

- SORT
- DeepSORT
- StrongSORT
- OC-SORT
- ByteTrack

## Planned Outputs

- Result video
- MOTChallenge format tracking txt
- TrackEval metrics
- Failure case snapshots
- Experiment config and environment info

## Architecture Direction

CLI and GUI execution must share the same core PaperPipeline / ExperimentRunner logic. The GUI should be a frontend wrapper only; tracking logic belongs in the reusable core package.

This initial repository setup contains documentation, configuration, and a minimal Python package skeleton. It does not implement the SORT algorithm yet.

## Current CLI

The current CLI supports paper-preset inspection and dry-run experiment folder creation only. It does not run tracking yet.

```powershell
python -m motlab.app.cli_main list-papers
python -m motlab.app.cli_main inspect-paper sort
python -m motlab.app.cli_main run --paper sort --dry-run
python -m motlab.app.cli_main run --paper sort --dry-run --output-root outputs/runs
```

The dry-run command creates a folder under `outputs/runs/` with:

- `paper_config.yaml`
- `environment.json`
- `run_manifest.json`
