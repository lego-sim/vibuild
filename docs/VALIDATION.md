# Validation

```bash
python -m pytest -q
VIBUILD_TEST_BLENDER=blender python -m pytest -q
./vibuild doctor
./vibuild check --record demo --hashes
```

Install `.[tools,test]` first. Blender tests skip unless `VIBUILD_TEST_BLENDER`
is set to an installed executable. Tests use temporary inputs and fake Codex
processes; they make no paid model requests. Test discovery excludes run artifacts.

Tests cover calibration, scoring, job cancellation, input integrity, frozen
assessment locks, atomic scene saves, scope, and continuous geometry. Passing
software tests does not establish scene quality or measured reconstruction accuracy.
