import copy
from pathlib import Path

import pytest

from vibuild.assessment.scoring import (
    RUBRIC,
    rubric_hash,
    validate_result,
    summarize_region,
    summarize,
    render_report,
)
from vibuild.agents.codex import validate_output_schema


def test_runtime_model_instructions_remain_english():
    import re
    from vibuild.assessment.scoring import prompt

    root = Path(__file__).parents[1]
    assert prompt({"region": "R01"}).isascii()
    assert "in English" in prompt({"region": "R01"})
    for path in list((root / "src/vibuild/agents/prompts").glob("*.md")):
        assert not re.search(r"[\u4e00-\u9fff]", path.read_text()), path


def fixture():
    packet = {
        "region": "S01",
        "name": "入口",
        "scene_sha256": "scene",
        "rubric_hash": rubric_hash(),
        "views": [{"id": "1"}, {"id": "2"}],
        "objects": [{"name": "wall"}],
    }
    result = {
        "region": "S01",
        "views": [
            {
                "id": fid,
                "scores": {d: 90 for d in RUBRIC["dimensions"]},
                "confidence": 0.9,
                "notes": "轮廓基本吻合",
            }
            for fid in ("1", "2")
        ],
        "findings": [],
        "uncertainties": [],
    }
    return packet, result


def finding(dimension="structure"):
    return {
        "dimension": dimension,
        "severity": "major",
        "frame_id": "1",
        "bbox": [0.1, 0.1, 0.4, 0.4],
        "objects": ["wall"],
        "missing_element": "",
        "observation": "开口位置与参考不符",
        "action": "按参考恢复开口",
    }


def test_fixed_aggregation_penalizes_worst_view_and_schema_valid():
    packet, result = fixture()
    result["views"][0]["scores"]["structure"] = 50
    result["findings"] = [finding()]
    region = summarize_region(result, packet)
    assert region["dimensions"]["structure"]["score"] == 64
    assert region["score"] == 82.2
    assert region["passed"]  # Aggregated 64 now clears the provisional dimension floor.
    validate_output_schema(
        Path(__file__).parents[1] / "src/vibuild/assessment/score.schema.json"
    )


def test_unknown_is_not_renormalized_into_pass():
    packet, result = fixture()
    result["views"][0]["scores"]["structure"] = None
    result["uncertainties"] = ["该视角墙体被遮挡"]
    region = summarize_region(result, packet)
    assert region["dimensions"]["structure"]["score"] == 31.5
    assert region["dimensions"]["structure"]["observed_score"] == 90
    assert region["score"] == 72.45
    assert region["observed_score"] == 90
    assert region["coverage"] == 0.85
    assert region["status"] == "needs_evidence"


def test_global_requires_every_region_and_fixed_scene():
    packet, result = fixture()
    a = summarize_region(result, packet)
    b = {
        **copy.deepcopy(a),
        "region": "S02",
        "passed": False,
        "status": "failed",
        "score": 74,
    }
    assert not summarize([a, b], ["S01", "S02"])["passed"]
    with pytest.raises(ValueError, match="all expected"):
        summarize([a], ["S01", "S02"])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([a, a])
    b["scene_sha256"] = "other"
    with pytest.raises(ValueError, match="mix scenes"):
        summarize([a, b])
    assert summarize([a], ["S01"])["passed"]
    assert not summarize([])["passed"]


def test_low_dimensions_require_localized_evidence():
    packet, result = fixture()
    result["views"][0]["scores"]["structure"] = 55
    with pytest.raises(ValueError, match="localized evidence"):
        validate_result(result, packet)
    result["findings"] = [finding()]
    assert validate_result(result, packet)


@pytest.mark.parametrize(
    "scores,passed",
    [
        ([75, 75, 75, 75, 75], True),
        ([70, 75, 75, 75, 75], False),
        ([90, 90, 90, 60, 90], True),
        ([90, 90, 90, 55, 90], False),
    ],
)
def test_provisional_region_and_dimension_boundaries(scores, passed):
    packet, result = fixture()
    for view in result["views"]:
        view["scores"] = dict(zip(RUBRIC["dimensions"], scores))
    if min(scores) < 60:
        result["findings"] = [finding("appearance")]
    assert summarize_region(result, packet)["passed"] is passed


@pytest.mark.parametrize("second_score,passed", [(85, True), (80, False)])
def test_provisional_global_boundary_requires_eighty(second_score, passed):
    regions = []
    for i, score in enumerate([75, second_score]):
        packet, result = fixture()
        packet["region"] = result["region"] = f"R{i + 1:02d}"
        for view in result["views"]:
            view["scores"] = {dimension: score for dimension in RUBRIC["dimensions"]}
        regions.append(summarize_region(result, packet))
    assert all(r["passed"] for r in regions)
    assert summarize(regions, ["R01", "R02"])["passed"] is passed


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda v: v["views"][0]["scores"].update(structure=91), "multiples"),
        (lambda v: v["views"][0]["scores"].update(structure=True), "multiples"),
        (lambda v: v["views"][0]["scores"].update(structure=float("nan")), "multiples"),
        (lambda v: v["views"][0].update(confidence=float("inf")), "confidence"),
        (lambda v: v["views"][0].update(id="other"), "frozen"),
        (lambda v: v["views"].append(copy.deepcopy(v["views"][0])), "frozen"),
        (lambda v: v.update(region="S02"), "region"),
        (lambda v: v["views"][0]["scores"].update(structure=None), "uncertainties"),
    ],
)
def test_invalid_evidence_rejected(mutate, pattern):
    packet, result = fixture()
    mutate(result)
    with pytest.raises(ValueError, match=pattern):
        validate_result(result, packet)


def test_findings_bind_pixels_and_exact_objects_and_critical_blocks():
    packet, result = fixture()
    finding = {
        "dimension": "structure",
        "severity": "critical",
        "frame_id": "1",
        "bbox": [0.1, 0.1, 0.4, 0.4],
        "objects": ["wall"],
        "missing_element": "",
        "observation": "开口被封堵",
        "action": "按参考恢复开口",
    }
    result["findings"] = [finding]
    region = summarize_region(result, packet)
    assert region["findings"][0]["priority"] == 4.5
    assert not region["passed"]
    finding["objects"] = ["invented"]
    with pytest.raises(ValueError, match="exact packet"):
        validate_result(result, packet)
    finding["objects"] = []
    with pytest.raises(ValueError, match="missing/unmapped"):
        validate_result(result, packet)
    finding["missing_element"] = "缺失门扇"
    assert validate_result(result, packet)
    finding["bbox"] = [0, 0, 0, 1]
    with pytest.raises(ValueError, match="bbox"):
        validate_result(result, packet)


def test_view_count_and_confidence_are_evidence_gates(tmp_path):
    packet, result = fixture()
    result["views"][0]["confidence"] = 0.6
    assert summarize_region(result, packet)["status"] == "needs_evidence"
    result["views"] = result["views"][1:]
    packet["views"] = packet["views"][1:]
    region = summarize_region(result, packet)
    assert region["status"] == "needs_evidence"
    summary = summarize([region])
    assert (
        "not measured depth" in render_report(tmp_path, summary, [region]).read_text()
    )


def test_report_overlays_reference_only_and_labels_incomplete(tmp_path):
    packet, result = fixture()
    packet["views"][0].update(width=640, height=360, card="view.jpg")
    result["findings"] = [finding()]
    region = summarize_region(result, packet)
    summary = summarize([region])
    summary.update(
        score=None, partial_score=90, missing_regions=["S02"], expected_regions=2
    )
    page = render_report(tmp_path, summary, [region]).read_text()
    assert 'viewBox="0 0 1280 424"' in page
    assert 'x="64.0" y="100.0" width="192.0"' in page
    assert "Overall 90" not in page
    assert "No global score yet" in page


def test_runtime_rubric_prompt_and_report_templates_are_english(tmp_path):
    import re
    from vibuild.assessment.scoring import prompt

    packet, result = fixture()
    packet["name"] = "Entrance"
    for view in result["views"]:
        view["notes"] = "Visible structure matches with minor defects."
    result["findings"] = []
    result["uncertainties"] = []
    region = summarize_region(result, packet)
    text = (
        str(RUBRIC)
        + prompt(packet)
        + render_report(tmp_path, summarize([region]), [region]).read_text()
    )
    assert not re.search(r"[\u3400-\u9fff]", text)


def test_unknown_view_blocks_acceptance_without_inflating_defect_priority():
    packet, result = fixture()
    result["views"][0]["scores"]["structure"] = None
    result["uncertainties"] = ["近景无法判断整体空间"]
    result["findings"] = [
        {
            "dimension": "structure",
            "severity": "major",
            "frame_id": "2",
            "bbox": [0.1, 0.1, 0.4, 0.4],
            "objects": ["wall"],
            "missing_element": "",
            "observation": "其余视角可见局部结构差异",
            "action": "核对墙边",
        }
    ]
    region = summarize_region(result, packet)
    assert region["dimensions"]["structure"]["score"] < 90
    assert region["findings"][0]["priority"] == 3
    assert region["evidence_requests"][0]["unknown_dimensions"] == ["structure"]
    assert not region["passed"]


def test_model_packet_omits_catalogue_but_preserves_scope_and_required_evidence():
    from vibuild.assessment.scoring import model_packet

    packet, _ = fixture()
    packet["objects"] = [
        {
            "name": "rail",
            "bounds": [0, 0, 0, 1, 1, 1],
            "materials": ["metal"],
            "faces": 12,
            "projected_bounds": {"1": [0, 0, 1, 1]},
        },
        {"name": "door"},
    ]
    packet["connection_requirements"] = [{"id": "I01", "components": ["floor"]}]
    packet["diagnostic_facts"] = [
        {"role": "shape", "source_sha256": "hash", "value": {"height": 2.5}}
    ]
    packet["views"][0]["card"] = "/full/path/image.jpg"
    value = model_packet(packet)
    assert "objects" not in value and "object_details" not in value
    assert value["views"] == ["1", "2"]
    assert value["connection_requirements"] == packet["connection_requirements"]
    assert value["diagnostic_facts"][0]["value"] == {"height": 2.5}
    assert "scene_sha256" not in value and "card" not in value["views"][0]
    assert packet["views"][0]["card"] == "/full/path/image.jpg"


def test_visual_findings_map_candidates_without_claiming_object_identity():
    packet, result = fixture()
    packet["objects"] = [
        {"name": "near_rail", "projected_bounds": {"1": [0.1, 0.1, 0.4, 0.4]}},
        {"name": "remote_rail", "projected_bounds": {"1": [0.7, 0.7, 0.9, 0.9]}},
        {"name": "background_wall", "projected_bounds": {"1": [0, 0, 1, 1]}},
    ]
    result["findings"] = [
        {
            "element": "Rail end",
            "dimension": "alignment",
            "severity": "major",
            "frame_id": "1",
            "bbox": [0.1, 0.1, 0.4, 0.4],
            "observation": "End deviates from source",
            "action": "Align the rail end to the source",
        }
    ]
    for view in result["views"]:
        view["scores"]["alignment"] = 50
    summary = summarize_region(result, packet)
    finding = summary["findings"][0]
    assert finding["candidate_objects"] == ["near_rail"]
    assert finding["objects"] == [] and finding["missing_element"] == "Rail end"
    assert not summary["passed"]
    # A source-visible omission remains a finding when no modeled object matches.
    packet["objects"] = []
    missing = summarize_region(result, packet)
    assert missing["findings"][0]["candidate_objects"] == []
    assert (
        missing["score"] == summary["score"] and missing["passed"] == summary["passed"]
    )


def test_visual_packet_uses_box_union_not_display_envelope():
    from vibuild.assessment.scoring import model_packet

    packet, _ = fixture()
    packet["scope"] = {
        "polygon": [[0, 0], [100, 100]],
        "core_z_interval": [0, 100],
        "core_boxes": [
            {"id": "A", "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]},
            {"id": "B", "bounds_min": [5, 5, 5], "bounds_max": [6, 6, 6]},
        ],
    }
    model = model_packet(packet)
    assert model["scope"]["core_boxes"] == packet["scope"]["core_boxes"]
    assert "polygon" not in model["scope"] and "core_z_interval" not in model["scope"]
