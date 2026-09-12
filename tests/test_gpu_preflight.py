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


def test_pro6000_profile_accepts_qualified_host():
    module_ = module()
    value = {
        "cuda_available": True,
        "gpu_count": 1,
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "total_memory_bytes": 96 * 1024**3,
                "capability": [12, 0],
            }
        ],
        "disk_free_bytes": 130 * 1024**3,
        "torch": "2.11.0+cu128",
        "cuda_runtime": "13.0",
    }
    assert module_.validate(value, profile="pro6000") == []


def test_pro6000_profile_rejects_old_torch_and_wrong_gpu():
    module_ = module()
    value = {
        "cuda_available": True,
        "gpu_count": 1,
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 3090",
                "total_memory_bytes": 24 * 1024**3,
                "capability": [8, 6],
            }
        ],
        "disk_free_bytes": 10 * 1024**3,
        "torch": "2.7.1+cu126",
        "cuda_runtime": "12.6",
    }
    errors = module_.validate(value, profile="pro6000")
    assert any("PRO 6000" in error for error in errors)
    assert any("90 GiB" in error for error in errors)
    assert any("12.0" in error for error in errors)
    assert any("2.11" in error for error in errors)
    assert any("20 GiB" in error for error in errors)


def test_default_profile_preserves_4x3090_contract():
    # Existing single-arg calls keep the legacy behavior.
    assert module().validate(report()) == []


def test_alex_8x_pro6000_profile_accepts_full_node():
    value = {
        "cuda_available": True,
        "gpu_count": 8,
        "gpus": [
            {
                "index": index,
                "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "total_memory_bytes": 96 * 1024**3,
                "capability": [12, 0],
            }
            for index in range(8)
        ],
        "disk_free_bytes": 500 * 1024**3,
        "torch": "2.11.0+cu128",
        "cuda_runtime": "12.8",
    }
    assert module().validate(value, profile="alex-8x-pro6000") == []
