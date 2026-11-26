# Auto-DDG Cutline

Auto-DDG Cutline is a companion to [DotDotGoose](https://github.com/khem-so-usfws/DotDotGoose) for counting seabirds in overlapping survey photos. It recommends a subset of photos and proposes paired cutlines that show where to hand off counting from one photo to the next. Review the drafts in HTML or DotDotGoose, adjust the photo selection in the launcher, and refine the cutlines at full resolution. DotDotGoose is where analysts can edit the exported lines and count birds.

The default COLMAP mode verifies image matches and exports editable cutlines and proposed keep sides in the original images' pixel coordinates.

## Windows setup

Open Anaconda Prompt in the companion repository. The existing environment and requirements are sufficient for the Python part.

```bat
conda activate auto-ddg-cutline
python -m pip install -r requirements.txt
```

To create the environment first, use `conda create -n auto-ddg-cutline python=3.12 pip`. Install `pytest` with `python -m pip install pytest` if you want to run the tests.

To open the launcher from source after setup, run `python -m auto_ddg_cutline.windows_launcher` from the repository folder.

Download a Windows x64 build from [COLMAP Releases](https://github.com/colmap/colmap/releases) and extract the full archive. Choose the CPU archive for a CPU-only workstation or the CUDA archive for NVIDIA acceleration. COLMAP 4.0.x is the reference CLI family. The wrapper checks subcommand help for the installed extraction and matching flags.

```bat
set "COLMAP_PATH=%USERPROFILE%\Tools\COLMAP\COLMAP.bat"
"%COLMAP_PATH%" feature_extractor -h
```

Adjust the path to your extracted archive. You can also set `COLMAP_PATH` to the extracted `colmap.exe`, pass `--colmap-path`, or have `colmap` on PATH. The cutline CLI has been tested on Windows with CPU and GPU COLMAP runs. For a packaged launcher, use the release build and smoke test below.

## Build a Windows launcher release

Open a 64-bit Windows Anaconda Prompt in the companion repo. The release ZIP contains the launcher and its Python runtime. Download and extract COLMAP separately.

```bat
conda activate auto-ddg-cutline
python -m pip install -r requirements-release.txt
build_windows.bat --version 0.1.6-overview.4
```

The batch script runs the tests, checks the packaged launcher and worker, then creates the ZIP and checksum under `dist\releases`. Use `--output-dir "C:\Temp\cutline-releases"` to choose another location.

1. Open `AutoDDGCutline.exe` and choose your extracted `COLMAP.bat`. The launcher remembers the path. **Test COLMAP** checks the installed version and local SfM support. **Test GPU** matches a known overlapping pair through the packaged worker.
2. Choose the colony folder. The main table shows every supported photo in filename order. **Analyze photos** uses the full folder as matching evidence and checks the recommended photos in **Use for refinement**. It favors locally sharper views at comparable scale on matched terrain. The counting target defaults to 25% of the folder, rounded down to a minimum target of two. Change or clear **Counting photo target (%)** under **Advanced**.
3. Open **Review drafts** to inspect the first paired cutline and its count sides. Draft routing mildly penalizes crossing strong color edges in either photo. Use the pair picker or **Previous** and **Next** to review more cutlines. **Export drafts to DDG…** appends the initial native-pixel lines to same-stem LabelMe files for inspection in DotDotGoose.
4. Back in the launcher, check or uncheck the photos you want to count. Search by filename or use **Checked only**. **Reset to recommended photos** restores the checkboxes from the most recent analysis or loaded recommendation. Click **Refine cutlines** to run on exactly the checked originals.
5. Open **Review refined cutlines** to inspect the seams at original pixel scale. Refinement rechecks the reviewed photo pairs that remain checked and tries to reconnect affected joins. The report flags added photos without a verified cutline. Local SfM targets unresolved handoffs.
6. Use **Export refined cutlines…** to save paired cutlines and keep sides for DDG. Existing annotations are preserved. To recover cutlines from a 0.1.4 job that produced unwanted polygons, export its saved review to a photos-only copy and inspect it in DDG.

To keep a bird group together, draw a loose polygon around the group in DDG after reviewing the draft. Select the polygon, open **Annotation Properties…**, and set its label to `protect_group`. Back in the launcher, check the photo and enable **Avoid DDG polygons labeled protect_group** before clicking **Refine cutlines**. The report shows the marked outlines and flags joins where a clear cutline could not be verified. Check which photo owns the group before exporting. Export into a photos-only copy to compare refined lines with earlier DDG drafts.

Use **Advanced** to choose saved matches, enable local SfM during the initial analysis, change the output folder, or configure benchmarking. **Open run logs** opens the folder containing the saved manifest and COLMAP logs. The packaged launcher shows the release version in its title bar.

To resume, choose the photo folder and click **Load previous selection…** above the table. An **Analyze photos** `run.json` or `selection-plan.json` restores the recommendation. A `tier2-selection.json` from a refinement workspace restores the analyst's exact checks and reopens the refined review when the completed job is still alongside it. If the original analysis plan moved, choose that exact file when prompted. **Reset to recommended photos** returns to the original software picks.

### Survey Overview

After **Analyze photos**, click **Update Survey Overview**. The default **Photos around the island** view groups real thumbnails by approximate camera direction. Select a **Focus photo** to highlight verified overlaps, then choose a neighbor to see both photos side by side. The **Survey Overview** link in the draft review opens the same report. Use **Checked photos** or **All photos** to switch between the counting set and the whole survey. Unregistered photos are flagged in the list. The overview works offline in a browser with Canvas support.

Change the checkboxes and click **Update Survey Overview** again to refresh the same HTML report next to the analysis plan. The overview reuses the cached alignment and terrain patches while the matching workspace and source photos are unchanged. After **Refine cutlines**, update the overview to see the refined handoffs. Keep the analysis job, including its staged images and verified match database, to use the overview again.

Switch to **Terrain (3D)** to inspect the geometry. Drag to rotate and scroll to zoom. Select **Photo A** and **Photo B** to compare their supported footprints. Cyan and pink show the two views, and gold shows shared support. Holes in the coarse surface mark areas with insufficient support. Turn off **Terrain** to see the sparse landmarks. Click a cutline to open the paired photo review and confirm the cutline and count sides. Use the reconstruction block picker when photos register in separate coordinate systems.

Turn on **Camera frustums** to see where the registered cameras were pointing and the angular extent of each photo. The toggle starts off, and the frustums follow the **All photos** or **Checked photos** filter. Use **Photo A** and **Photo B** to highlight their frustums in cyan and pink. Use the colored patches on the terrain to see supported overlap.

To try the dense terrain pilot, click **Build dense terrain…** in the launcher and pick a registered photo of the section you want to inspect. The reference needs at least two other registered views with shared tracks in the same block. This step requires the CUDA build of COLMAP and an NVIDIA GPU. It uses the saved camera poses, up to eight views, and a 1,400-pixel long edge to make a photo-colored surface for the reference area. The report opens with **Dense terrain** selected. Switch to **Sparse terrain** for the whole block. The dense layer displays footprints and cutline stretches with depth support in that section. The paired photo review and DDG export still use the original pixel coordinates and saved cutlines. **Update Survey Overview** reuses a successful dense pilot with your current photo checks.

Choose **Auto** for GPU matching with visible CPU fallback. **NVIDIA GPU** runs the matching stages on the GPU alone. **CPU** uses CPU matching. The report and run logs let you check the result, and the status line shows which device supplied the matches. Local SfM uses the CPU mapper. Use a CUDA build with **Test GPU** on the workstation that will run the analysis.

During setup, the launcher prints the COLMAP executable and each help check. A check that exceeds 30 seconds ends with the relevant log path.

## Generate paired cutlines

A photos-only copy is useful if the earlier experiment wrote sparse regions beside your originals. JPEG, PNG, TIFF, and BMP inputs are read from the selected folder.

```bat
python -m auto_ddg_cutline "C:\survey\colony-test" --list
python -m auto_ddg_cutline "C:\survey\colony-test" --select 1-2 --device cpu
python -m auto_ddg_cutline "C:\survey\colony-test" --select 1-2 --device cpu --write
```

Use comma-separated ranges or numbers such as `--select 2-4,7` for nonconsecutive images. Use `--all` if the folder is already curated. The tool uses selection order to propose neighboring handoffs and evaluates skip-one overlaps for extra boundaries. `--device auto` tries accelerated feature extraction and matching, then retries on CPU when it detects a device-related failure. `--device gpu` requires acceleration.

CPU SIFT extraction defaults to one worker, including after a GPU-to-CPU retry. Full-resolution SIFT workers each need memory for their own image pyramid. Use `--cpu-threads N` to choose a different worker count. The image dimensions and feature settings stay the same.

For paired cutlines, auto mode also retries on CPU if a completed GPU run yields zero usable verified handoff pairs. The terminal reports how many photos produced descriptors, how many pairs matched before verification, and how many pairs the cutline planner can use. The `matching_attempts` entries in `run.json` retain those diagnostics for each completed attempt. Use `--device gpu` to diagnose the GPU path by itself.

Each run uses a fresh job workspace. The terminal prints the path to `run.json`, which records the selected dimensions, verified matches, handoffs, keep sides, and sidecar changes. Use `--work-dir "C:\survey\cutline-jobs"` to keep the workspaces together. Include `--write` to append the proposed lines to same-stem LabelMe files. The job folder contains temporary photo copies, the COLMAP database, and logs.

If COLMAP exits during extraction, check the reported exit code and `cpu\extract.log` in the new job folder. The manifest records the failed subcommand, exit status, and log path.

The tool stages decoded native pixel grids as PNGs in their stored orientation. COLMAP extracts at the native long edge by default. The cutline planner searches for a continuous route with endpoints along the shared image boundaries. It uses image agreement and nearby landmarks, with local alignment when supported. If the movable route fails a check, the planner retries with fixed endpoints. The terminal and `run.json` report which endpoint search was used and distinguish route-search, clipping, and count-side failures. A straight midpoint fallback remains a `DRAFT`.

## Benchmark and reuse colony runs

In **Advanced**, choose a preset or type a custom number in **Fresh extraction (% of original pixels)**. Fresh **Analyze photos** runs default to **50%**. For a 5616 × 3744 photo, 50%, 53.2%, and 30% correspond to approximately 3971, 4096, and 3076 pixels on the long edge. When using saved matches, the launcher displays the saved extraction percentage or the recorded long-edge limit. Clear **Reuse matches** to apply a new extraction size. For comparisons, use the same matching device and compare `run.json` and `selection-plan.json` across runs.

**CPU extraction workers** defaults to **1** and is editable while the launcher is idle. It applies when **Analyze photos** or **Refine cutlines** extracts features on the CPU, including an automatic fallback to CPU. **Refine cutlines** extracts from the checked originals at native resolution.

```bat
python -m auto_ddg_cutline "C:\Temp\219-055-full" --all --output selection-plan --device auto --feature-pixel-percent 100 --work-dir "C:\Temp\cutline-bench"
python -m auto_ddg_cutline "C:\Temp\219-055-full" --all --output selection-plan --device auto --feature-pixel-percent 30 --work-dir "C:\Temp\cutline-bench"
```

The percentage is applied to each photo's pixel count, with its aspect ratio preserved. The staged PNG grid, cutline coordinates, and DDG sidecar dimensions stay at native resolution. The existing `--feature-max-edge` option also accepts an absolute long-edge limit. `run.json` records timings, matching device, extraction limits, and verified overlap counts. Add `--profile-resources` for sampled peak process-tree RAM when `psutil` is installed and sampled device-wide GPU memory when `nvidia-smi` is available. Add `--verbose` for more detailed COLMAP progress.

```text
[00:00:00] staging started | stage 0.0s
[00:00:04] staging progress 10/93 | stage 4.0s | photos staged
[00:02:17] matching and verification (GPU) started | stage 0.0s
[00:02:47] matching and verification (GPU) heartbeat | stage 30.0s | still running
```

Keep the original job's `run.json`, native staged `images` directory, and the successful device folder's `database.db` together. Reuse supports compatible completed or partial movable-endpoints runs. Use the same photos in the same order. The new job validates the inputs before using the saved matches. The source job's matching settings determine the feature size and device. Specify device and feature settings when starting a fresh run.

```bat
set "RUN=C:\Temp\cutline-bench\auto-ddg-colmap-REPLACE-ME\run.json"
python -m auto_ddg_cutline "C:\Temp\219-055-full" --all --reuse-matches "%RUN%" --work-dir "C:\Temp\cutline-bench"
python -m auto_ddg_cutline "C:\Temp\219-055-full" --all --reuse-matches "%RUN%" --output sfm-diagnostics --verbose --work-dir "C:\Temp\cutline-bench"
```

The cutline reuse run reads the saved match database directly. SfM diagnostics work in a new job with a SQLite copy of that database. The diagnostics report tracks, registration membership, per-photo reprojection errors, and camera pairs separated by at most `0.00001` of the model's largest camera baseline. Treat those flags as a cue to inspect the camera poses. `sfm-diagnostics` is analysis only and takes at least three photos.

## Review a full colony photo selection

Set `PHOTOS` to the full colony folder and `RUN` to its saved COLMAP run. The same photo names, order, and source files must be available. The matching job must still have its database and staged images. Omit `--reuse-matches` to match the folder from scratch. Add `--verbose` for more frequent progress updates or `--profile-resources` for sampled resource use.

```bat
set "PHOTOS=C:\Temp\219-055-full"
set "RUN=%TEMP%\auto-ddg-colmap-m0bt58ah\run.json"
python -m auto_ddg_cutline "%PHOTOS%" --all --reuse-matches "%RUN%" --output selection-plan --work-dir "C:\Temp\cutline-selection"
```

Open the `selection-plan.html` path printed at the end of the run. Start with flagged joins and possible omissions, then check the counting order and expand the paired cutline previews. Yellow lines show draft cutlines and green rings show the intended count side. Where a pair lacks a draft, numbered landmarks show the available overlap evidence. Orange cells on photo cards show areas without distributed matches. Review those areas in the original photos for unique terrain, clearer birds, and occlusions before confirming whole-colony coverage.

To pin or exclude the same photos across analysis runs, save a rules file and choose **Load pin/exclude rules…** under **Advanced** before running **Analyze photos**. In the CLI, use `--selection-rules "C:\Temp\219-055-rules.json"`. Use the original filenames exactly as shown in the report. For a single review, edit the photo checkboxes before refinement.

```json
{
  "keep": ["2021_06_09_Detail_219-055_IMG_2104.JPG"],
  "exclude": [],
  "exclusion_reasons": {}
}
```

The report records gaps if a rule excludes the only eligible view of a documented area. `selection-plan.json` saves the photo IDs, hashes, handoff routes, draft lines and keep points in stored source pixels. Use `run.json` for stage timings and review counts. Use `--count-selection 2,4-6` with `--all --output selection-plan` to review a fixed counting set while retaining the whole folder as overlap evidence. The launcher can export the lines directly from the reviewed result.

Add `--target-photo-percent 25` to the saved-match selection command to check the rest of the verified overlap pairs and compare compact selection with the full documented-coverage proposals. Check sections with no counting photo, documented gaps and suggested bridges in the report before accepting the selection. Well-supported unique terrain, separate documented sections and analyst pins can take the proposal above the target.

### Generate drafts from local SfM

Add `--run-local-sfm` to run the suggested local blocks using the same verified matches. The installed COLMAP mapper must support `--Mapper.image_list_path` or `--image_list_path`. The tool copies the match database once and reuses it across the local blocks.

```bat
python -m auto_ddg_cutline "%PHOTOS%" --all --reuse-matches "%RUN%" --output selection-plan --run-local-sfm --work-dir "C:\Temp\cutline-selection"
```

The report shows which views registered together, the number of shared 3D landmarks, and any failed handoffs. The tool projects validated landmarks into both source photos and uses them to fit paired terrain-following lines. Before a draft can pass, the pair needs at least 16 shared landmarks, including eight supported by a third view. Each proposed line and count side is still a draft for review. Add `--write` to append only the local SfM drafts to DDG sidecars. Existing paired cutlines are preserved. Local block support does not establish full colony coverage.

For colony 219-055, compare the recommendation with the analyst's 15-photo set. Check whether a bridge connects 2079 and 2090, whether 2062 and 2063 can support a usable cutline, and whether 2104 contributes countable terrain beyond 2103 and 2105. Record the recommended photo count, sections, coverage gaps, unresolved handoffs, three-view flags, elapsed stages, and review time.

### Review the output in DDG

| Terminal status | Meaning |
| --- | --- |
| `PROPOSED` | Both photos have a continuous cutline and a proposed keep side in their shared overlap. |
| `DRAFT` | A pair passed the line and keep-side checks, but alignment or an overlapping third photo warrants review. |
| `Unresolved` | An overlap or its keep side could not be established for this pair. |
| `PARTIAL` | At least one neighboring pair is unresolved or a proposed line needs extra review. |

Open the paired photos together in DDG and check that the lines cross the same physical landmarks. The terminal and `run.json` tell you which side to keep in each photo's overlap. A newly exported line has a shared pair identifier, its partner filename, and a keep point chosen beside the line. To form a count polygon later, use the handoff to split a count-area outline and keep the indicated side. Check where the cutline meets the outline and extend it where necessary. The DDG export in this mode consists of editable line annotations.

Writing appends to existing sidecars and keeps earlier shapes and DDG edits. If a linked pair is already present, the exporter reports `KEEP` for that pair. To review new proposals for the same photos, run with `--write` in a photos-only copy of the folder.

## Compare with the OpenCV baseline

```bat
python -m auto_ddg_cutline "C:\survey\baseline-copy" --select 1-3 --backend opencv --method terrain --write
```

The OpenCV backend retains the terrain, image, and midpoint methods and the optional `--review-dir` PNG from Milestone 1. Use separate image copies when comparing annotations. The earlier COLMAP sparse-region experiment is available with `--output sparse-regions` and at least three photos.

## Run the tests

```bat
python -m pytest -v
```

The tests cover verified match decoding, paired clipping and keep sides, annotation preservation, device fallback, and both backends. On the three sample cliff photos, the COLMAP cutline mode produces two neighboring handoffs.

## Generative AI disclosure

During November 2026, OpenAI ChatGPT and Codex (GPT-6 model family) helped
generate and revise Python code, automated tests, and documentation for this
companion, including photo selection, cutline refinement, and the Windows
launcher. Automated tests check matching, selection, refinement, and export.
The project maintainer is responsible for reviewing changes, validating the
Windows release, and approving the released software.

Any use of trade, firm, or product names is for descriptive purposes only and
does not imply endorsement by the U.S. Government.
