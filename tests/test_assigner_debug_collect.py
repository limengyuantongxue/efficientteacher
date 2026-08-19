#!/usr/bin/env python
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.assigner.scale_aware_assigner import ScaleAwareAssigner

def test_debug_collect_basic():
    print("=== Test: Basic debug collection ===")
    assigner = ScaleAwareAssigner(
        nc=80, na=3, nl=3,
        anchors=[[[10,13],[16,30],[33,23]], [[30,61],[62,45],[59,119]], [[116,90],[156,198],[373,326]]],
        stride=[8, 16, 32],
        top_k=13,
    )

    # Enable debug collection
    assigner.enable_debug_collect()
    assert assigner._collect_debug is True

    # Simulate predictions: 3 layers, batch=2, na=3, each layer dims
    # Layer 0: 80x80, Layer 1: 40x40, Layer 2: 20x20
    p = [
        torch.randn(2, 3, 80, 80, 85 + 1),   # (B, na, ny, nx, no)
        torch.randn(2, 3, 40, 40, 85 + 1),
        torch.randn(2, 3, 20, 20, 85 + 1),
    ]
    # targets: [batch_idx, cls, x, y, w, h] normalized
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.3],   # large box on image 0
        [0, 1, 0.3, 0.3, 0.05, 0.06], # small box on image 0
        [1, 2, 0.7, 0.6, 0.15, 0.2],  # medium box on image 1
    ])

    result = assigner(p, targets, img_size=640)
    assert result is not None, "forward() should return result"
    assert len(result) == 6, "Should return 6 items"

    debug_data = assigner.get_debug_data()
    assert len(debug_data) > 0, "Should have collected debug data"
    print(f"  Collected {len(debug_data)} records")

    for rec in debug_data:
        assert "area" in rec
        assert "dynamic_ks_iou" in rec
        assert "dynamic_ks" in rec
        assert "alpha" in rec
        assert isinstance(rec["dynamic_ks_iou"], int)
        assert isinstance(rec["dynamic_ks"], int)
        assert rec["dynamic_ks"] >= 1
        assert rec["dynamic_ks_iou"] >= 1

    print("  All fields present and valid")
    print("  PASSED")

def test_disable_clears():
    print("=== Test: Disable and clear ===")
    assigner = ScaleAwareAssigner(
        nc=80, na=3, nl=3, anchors=[[[10,13],[16,30],[33,23]]], stride=[8], top_k=13,
    )
    assigner.enable_debug_collect()
    p = [torch.randn(2, 3, 80, 80, 86)]
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
    assigner(p, targets)
    data_before = assigner.get_debug_data()
    assert len(data_before) > 0

    assigner.disable_debug_collect()
    assigner.clear_debug_data()
    assert len(assigner.get_debug_data()) == 0

    # Run again with debug off - should NOT collect
    assigner(p, targets)
    assert len(assigner.get_debug_data()) == 0, "Should not collect when disabled"

    print("  PASSED")

if __name__ == "__main__":
    test_debug_collect_basic()
    test_disable_clears()
    print("\nAll tests passed!")
