# Inputs

Use `vibuild tool vidmap run --input VIDEO --output work/registration --device cuda`
for camera registration. The pinned upstream checkout writes its COLMAP reconstruction
to `rec/`. Export/review the registered scene transform, RGB/timestamp associations,
and camera matrices below before freezing them; this alignment/export is a separate
preparation step, not an implicit part of `vibuild task`.

`vibuild init --record demo --videos VIDEO_DIRECTORY` only copies original videos.
Before modeling, supply:

```text
records/demo/record.json
records/demo/resources/index.json
records/demo/resources/alignment/frames.json
records/demo/resources/alignment/cameras.json
records/demo/resources/alignment/coordinates.json
records/demo/resources/input/...             original videos and RGB frames
partition.json                              reviewed scope and evidence associations
initial.blend                               packed scene with matching cameras
```

`record.json` identifies the record. `index.json` maps `frames`, `cameras`, and
`coordinates` to paths relative to resources, and optionally lists `videos`.
Frame entries contain `name`, numeric `index`, `video`, `time`, `width`, `height`,
3x3 `K`, 3x3 camera-to-scene `rotation`, 3-vector `center`, `original_rgb` relative
to resources, and `evaluation_split` (`fit` or held-out `audit`). Pose uses OpenCV
axes; Blender cameras use `rotation @ diag(1,-1,-1)`. Coordinates record the fixed
basis, origin, and scale; never infer metric accuracy from registration alone.

The reviewed partition contains:

- `schema_version`: `partition_v1`.
- `regions`: IDs such as R01, names, `bounds_min`, `bounds_max`, colors, and optional
  `core_boxes` unions. Bounds use the registered scene coordinate system.
- `frames`: calibrated frame names, `core_region`, and `visible_regions` entries with
  `region` and `fraction_of_registered_observations`. Views must actually show their
  assigned space; ownership of the camera alone is insufficient.
- `interfaces`: unique IDs, endpoint `regions`, `status: observed_transition`, finite
  bounds/patches, and source crossing evidence. Unknown edges remain unknown.
- `provenance.source_sha256`: hashes of resource-relative frame and coordinate files.
- Optional `topology.start_region` and reviewed `execution_plan`. The planner derives
  serial dependencies for chains/simple cycles; branching or complex cycles require
  an explicit reviewed plan with ordered `chains[].steps`.

Each step has an Rxx/Lxx/Jxx ID, kind, scope, ownership, and dependencies. Region
records supply local bounds; closure/connection stages name involved regions and
interfaces. The initial scene may have empty geometry but must already contain the
registered cameras and required scene settings. Keep one task-root scene thereafter.

`examples/indoor_partition.py` is an opt-in recipe for one capture, including its
reviewed bounds and topology. It expects aligned vidmap/COLMAP resources; edit it
for another capture. It is not an automatic universal segmentation tool.
