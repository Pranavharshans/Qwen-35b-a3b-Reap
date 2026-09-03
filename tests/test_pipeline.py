import json

from reverse_reap.pipeline import analyze_telemetry


def test_analysis_pipeline_writes_complete_candidate_bundle(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    rows = []
    for domain in ("coding", "control"):
        for sample in range(6):
            for expert in range(4):
                value = (8 - expert) if domain == "coding" else (expert + 1)
                rows.append(
                    {
                        "sample_id": f"{domain}-{sample}",
                        "domain": domain,
                        "stratum": "shared",
                        "split": "calibration" if sample < 3 else "selection",
                        "segment": "joint",
                        "layer": 0,
                        "expert": expert,
                        "routed_count": 2,
                        "reap_saliency": value + sample * 0.001,
                    }
                )
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "analysis"
    report = analyze_telemetry(
        telemetry,
        output,
        top_n=1,
        bootstrap_iterations=10,
        permutation_iterations=10,
        seed=5,
    )
    assert report["experts_ranked"] == 4
    ranking = json.loads((output / "expert-ranking.json").read_text())
    assert "differential_bootstrap_95ci" in ranking[0]
    assert "label_permutation_p_value" in ranking[0]
    assert "coding_routing_count" in ranking[0]
    for filename in (
        "expert-ranking.json",
        "bootstrap-stability.json",
        "label-permutation.json",
        "candidate-manifest.json",
        "control-manifests.json",
    ):
        assert (output / filename).exists()
    assert len(list((output / "controls").glob("frequency-random-*.json"))) == 20
    assert (output / "controls" / "frequency-matched.json").exists()
    assert (output / "controls" / "highest-frequency.json").exists()
