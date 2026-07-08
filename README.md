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

The current CLI supports paper-preset inspection, dry-run experiment folder creation, and MOT public detection based SORT execution.

```powershell
python -m motlab.app.cli_main list-papers
python -m motlab.app.cli_main inspect-paper sort
python -m motlab.app.cli_main run --paper sort --dry-run
python -m motlab.app.cli_main run --paper sort --dry-run --output-root outputs/runs
python -m motlab.app.cli_main run-sort-mot --detections tests/fixtures/mot/det.txt --output .tmp/sort_results.txt
python -m motlab.app.cli_main run-sort-mot --detections tests/fixtures/mot/det.txt --output-root outputs/runs --as-run-folder
python -m motlab.app.cli_main inspect-mot-sequence --sequence-dir datasets/MOT17/train/MOT17-02-FRCNN
python -m motlab.app.cli_main run-sort-sequence --sequence-dir datasets/MOT17/train/MOT17-02-FRCNN --output-root outputs/runs
python -m motlab.app.cli_main export-trackeval-layout --run-dir outputs/runs/<run_id> --sequence-name MOT17-02 --output-root outputs/trackeval
python -m motlab.app.cli_main check-trackeval --trackeval-root third_party/TrackEval
python -m motlab.app.cli_main prepare-trackeval --trackeval-root third_party/TrackEval
python -m motlab.app.cli_main prepare-trackeval --trackeval-root third_party/TrackEval --clone
python -m motlab.app.cli_main build-trackeval-command --trackeval-root third_party/TrackEval --gt-folder datasets/MOT17/train --trackers-folder outputs/trackeval/sort/<run_id>/trackers --seqmap-file outputs/trackeval/sort/<run_id>/seqmaps/MOT17-test.txt --tracker-name sort
python -m motlab.app.cli_main run-trackeval --trackeval-root third_party/TrackEval --gt-folder datasets/MOT17/train --trackers-folder outputs/trackeval/sort/<run_id>/trackers --seqmap-file outputs/trackeval/sort/<run_id>/seqmaps/MOT17-test.txt --tracker-name sort --output-dir outputs/trackeval_logs/<run_id>
```

The dry-run command creates a folder under `outputs/runs/` with:

- `paper_config.yaml`
- `environment.json`
- `run_manifest.json`

The `run-sort-mot` command currently supports SORT execution from MOTChallenge public detections only. It does not execute Faster R-CNN and does not run TrackEval.

When `run-sort-mot` is executed with `--as-run-folder`, it creates a timestamped run folder containing:

- `tracks.txt`
- `paper_config.yaml`
- `environment.json`
- `run_manifest.json`

## MOTChallenge Sequence Execution

The project can inspect and run SORT on an already downloaded MOTChallenge sequence folder. Dataset download is not automated.

```powershell
python -m motlab.app.cli_main inspect-mot-sequence --sequence-dir datasets/MOT17/train/MOT17-02-FRCNN
python -m motlab.app.cli_main run-sort-sequence --sequence-dir datasets/MOT17/train/MOT17-02-FRCNN --output-root outputs/runs
```

`run-sort-sequence` reads the sequence metadata from `seqinfo.ini` and uses `det/det.txt` public detections. It does not execute Faster R-CNN, YOLO, or TrackEval.

The generated run folder follows the same SORT MOT run structure:

- `tracks.txt`
- `paper_config.yaml`
- `environment.json`
- `run_manifest.json`

The manifest records sequence metadata such as sequence name, sequence directory, sequence length, frame rate, and image size.

## TrackEval Layout Export

The project can export an existing SORT MOT run folder into a TrackEval-friendly result layout. This step only prepares files for a future TrackEval run; it does not download or execute TrackEval.

```powershell
python -m motlab.app.cli_main export-trackeval-layout --run-dir outputs/runs/<run_id> --sequence-name MOT17-02 --output-root outputs/trackeval
```

`<run_id>` is a placeholder. Replace it with the actual folder name printed by `run-sort-mot`.

The exported layout is:

```text
outputs/trackeval/sort/<run_id>/
  trackers/
    sort/
      data/
        MOT17-02.txt
  seqmaps/
    MOT17-test.txt
```

`MOT17-02.txt` is copied from the run folder's `tracks.txt`. The seqmap contains a `name` header and the selected sequence name.

## TrackEval Preparation

The current TrackEval support is a pre-execution wrapper/check stage only. It does not download MOT17 ground truth and does not run quantitative evaluation yet.

`check-trackeval` only checks whether TrackEval is available. It never clones or installs TrackEval.

```powershell
python -m motlab.app.cli_main check-trackeval --trackeval-root third_party/TrackEval
```

The status line reports whether the root is missing, available, or found with a failed help/import check.

TrackEval can be prepared only when the user explicitly runs `prepare-trackeval --clone`.

```powershell
python -m motlab.app.cli_main prepare-trackeval --trackeval-root third_party/TrackEval
python -m motlab.app.cli_main prepare-trackeval --trackeval-root third_party/TrackEval --clone
```

Without `--clone`, `prepare-trackeval` prints the explicit clone guidance and writes setup metadata only. With `--clone`, it clones `https://github.com/JonathonLuiten/TrackEval.git` at `master` into `third_party/TrackEval` if that directory does not already exist.

`third_party/TrackEval` is ignored by Git and must not be committed. After preparation, run:

```powershell
python -m motlab.app.cli_main check-trackeval --trackeval-root third_party/TrackEval
```

The command builder prints a candidate TrackEval MOTChallenge command without executing it:

```powershell
python -m motlab.app.cli_main build-trackeval-command --trackeval-root third_party/TrackEval --gt-folder datasets/MOT17/train --trackers-folder outputs/trackeval/sort/<run_id>/trackers --seqmap-file outputs/trackeval/sort/<run_id>/seqmaps/MOT17-test.txt --tracker-name sort
```

The output includes a readable multi-line command for option review and a copyable one-line command.

This is an initial wrapper. TrackEval options must be validated against the local TrackEval checkout before a full evaluation workflow is enabled. Result parsing is planned for a later step.

## TrackEval Execution Wrapper

`run-trackeval` builds the MOTChallenge TrackEval command and stores execution logs. By default it is a dry-run and does not execute TrackEval.

```powershell
python -m motlab.app.cli_main run-trackeval --trackeval-root third_party/TrackEval --gt-folder datasets/MOT17/train --trackers-folder outputs/trackeval/sort/<run_id>/trackers --seqmap-file outputs/trackeval/sort/<run_id>/seqmaps/MOT17-test.txt --tracker-name sort --output-dir outputs/trackeval_logs/<run_id>
```

Dry-run creates:

- `command.txt`

Actual execution happens only when `--execute` is provided:

```powershell
python -m motlab.app.cli_main run-trackeval --trackeval-root third_party/TrackEval --gt-folder datasets/MOT17/train --trackers-folder outputs/trackeval/sort/<run_id>/trackers --seqmap-file outputs/trackeval/sort/<run_id>/seqmaps/MOT17-test.txt --tracker-name sort --output-dir outputs/trackeval_logs/<run_id> --execute
```

Execution stores:

- `command.txt`
- `stdout.txt`
- `stderr.txt`

If the MOT17 GT dataset path is not prepared, real evaluation can fail. This stage records command/stdout/stderr only; metric parsing is planned for a later step.

## MOTChallenge Format Support

The project currently supports MOTChallenge public detection loading and MOTChallenge tracking result writing.

Supported now:

- Load public detection `det.txt` rows in `frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z` format.
- Group detections by 1-based frame index.
- Write tracking results as `frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1`.
- Validate 1-based frame indices and positive tlwh bounding box sizes in shared data models.

Not implemented yet:

- Faster R-CNN detector execution or download
- TrackEval execution or download

## SORT Implementation Status

Implemented:

- MOTChallenge input/output utilities
- MOT data model validation
- IoU utilities for tlwh bounding boxes
- Hungarian association utility for IoU matching
- Kalman motion model wrapper
- SORT single-track lifecycle object
- Full frame-level multi-object SortTracker
- MOT public detection based SORT pipeline
- MOTChallenge sequence adapter and sequence run CLI

Not yet implemented:

- TrackEval metric parsing and validated full evaluation workflow
- Faster R-CNN execution
- GUI
