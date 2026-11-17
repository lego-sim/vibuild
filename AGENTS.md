# Contributing

Keep the public workflow in README.md and docs/workflow.md usable from an installed
wheel outside the checkout. Runtime prompts live in src/vibuild/agents/prompts;
scoring policy lives in src/vibuild/assessment/policy.json. Do not duplicate them here.

Run `python -m pytest`. Real Blender tests use `VIBUILD_TEST_BLENDER`. Use temporary
projects for tests. Existing records and run resources are frozen inputs. Keep runtime
code in its domain package; add no checkout-only wrappers or compatibility aliases.
