import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / "scripts" / "gpu_preflight.py"
    spec = importlib.util.spec_from_file_location("gpu_preflight", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def report(name="NVIDIA GeForce RTX 3090", count=4, memory=24 * 1024**3):
    return {
        "cuda_available": True,
        "gpu_count": count,
        "gpus": [
            {
                "index": index,
                "name": name,
                "total_memory_bytes": memory,
                "capability": [8, 6],
            }
            for index in range(count)
        ],
        "disk_free_bytes": 250 * 1024**3,
    }


def test_accepts_exact_4x3090_contract():
    assert module().validate(report()) == []


def test_rejects_wrong_count_model_memory_and_disk():
    value = report(name="Tesla V100", count=2, memory=16 * 1024**3)
    value["disk_free_bytes"] = 100
    errors = module().validate(value)
    assert any("exactly 4" in error for error in errors)
    assert any("not an RTX 3090" in error for error in errors)
    assert any("23 GiB" in error for error in errors)
    assert any("100 GiB" in error for error in errors)
