# Visual scoring

```bash
./vibuild tool score prepare --record demo --partition partition.json \
  --scene TASK_ROOT/scene.blend --region R01 --output TASK_ROOT/R01/scores/001
./vibuild tool score run --output TASK_ROOT/R01/scores/001
./vibuild tool score report --output TASK_ROOT/R01/scores/001
```

Preparation fixes source views, calibrated renders, scope, and structural evidence.
Each assessor is a fresh read-only session receiving only its compact packet and
images. It neither edits geometry nor renders or runs depth inference. Model and
reasoning settings come from the tool config; CLI overrides are optional.

| Dimension | Weight | Visible evidence |
| --- | ---: | --- |
| Structure | 30% | Walls, floors, ceilings, openings, and spatial layout |
| Completeness | 25% | Required visible components and their parts |
| Alignment | 20% | Position, scale, orientation, and silhouette |
| Appearance | 15% | Color, texture scale, and material response |
| Connectivity | 10% | Continuity, contact, and visible intersections |

Per-view scores use 0–100 in steps of 5; unknown is `null`. Anchors: 100 no visible
difference, 80 minor defects, 70 clear local defects, 50 major errors, 25 fragmentary
agreement, 0 missing or contradictory. Regional dimension score is 70% view mean
plus 30% worst view. Unknown evidence contributes zero to the conservative score.

Local passage requires total >=75, every dimension >=60, at least two useful views,
confidence >=0.7, required evidence known, and no critical defects. Global passage
requires an equal-region mean >=80, all regions passing, and required connection
and held-out checks. Connection scores are separate gates, not extra regions in the mean.

Findings identify visual elements, frame IDs, image boxes, severity, and suggested
repairs. The program maps spatial object candidates; these do not grant ownership.
Prioritize missing major structure before appearance. Low-score priority uses weight,
observed deficit, and severity; unknowns request evidence instead of invented repairs.

Choose non-audit views by source visibility before scoring. Never drop a difficult
view to raise the score. Off-screen content is not a modeled omission. Connection
assessments retain all required interface views and component checks. Scores cover
visible evidence, not hidden geometry or physical accuracy. Evidence/rubric/model
changes invalidate cached assessments. Apply [bounded closeout](../PLAN.md#closeout)
after visual passage; seam-only corrections do not restart full scoring.
