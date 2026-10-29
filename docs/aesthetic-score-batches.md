# Aesthetic scoring: single image or ZIP batch

Submit `type: "aesthetic-score"` to `/v1/jobs`.

Existing single-image params (`image` base64 or `image_file` staged path) and
result shape are unchanged: `scores`, `elapsed_ms`, `model_repository`,
`score_names`.

For batches, stage a ZIP of **original image files** with
`arbiter_client.stage_file(path)` and submit `params.images_file` containing the
returned Spark path. Do not combine it with `image` or `image_file`. There is no
base64 ZIP parameter. `images_file` passes the same generic `*_file` submission
validation, content hashing, and inbox lifecycle as other staged inputs. It is
not extracted to disk. Normal staged-file lease/orphan cleanup applies; completion
does not immediately delete the ZIP (shared-input grace periods are preserved).

Use **32–64 original files per client job** to stay under the deployed model's
60-second max runtime. The adapter preprocesses one original at a time into small
CPU tensors, then runs real GPU forward passes of up to **32 images**. This
bounds both original-image memory and GPU tensor batches. Image decoding time
still counts toward runtime; unusually large/slow originals may need smaller
client ZIPs. The adapter processes all members; there is no silent truncation.

Batch results replace the single `scores` field with `items`:

```json
{
  "items": [
    {"name": "originals/photo.jpg", "scores": {
      "overall_aesthetic": 3.1, "technical_quality": 3.2,
      "composition": 3.3, "lighting": 3.4, "color_harmony": 3.5,
      "depth_of_field": 3.6, "content": 3.7
    }},
    {"name": "broken.heic", "error": "bad input image: ..."}
  ],
  "elapsed_ms": 123.4,
  "model_repository": "rsinema/aesthetic-scorer",
  "score_names": ["overall_aesthetic", "technical_quality", "composition",
    "lighting", "color_harmony", "depth_of_field", "content"]
}
```

Numbers above illustrate the shape, not measured scores. Scores are raw model
outputs, not normalized percentages. Items follow ZIP central-directory order;
directories are omitted. Duplicate names remain separate items in that order.
Every file member (including resource forks/non-images) gets either all seven
finite scores or an error, never an omitted result. Corrupt/empty ZIPs fail the
job. A failed GPU forward marks that minibatch's members with errors; cancellation
still aborts the job. Metadata and JSON result-file writing match single mode.

Verification:

- `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_aesthetic_batch.py`
- After deployment, run the real integration test **detached**, then inspect its
  verdict: `PYTHONPATH=src .venv/bin/python -m pytest -s tests/integration/test_aesthetic_batch.py`.
  It stages 64 ordered members (63 JPEGs, one corrupt member), crosses the
  32-image GPU boundary, checks the partial final minibatch, and compares the
  first image to a forced legacy single-image job using the real model.
