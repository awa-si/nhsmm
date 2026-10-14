import json

import numpy as np

from scripts.compare_optimizer_bridge import PROTOCOL, run_seed


def test_real_bounded_bridge_smoke():
    contract = json.loads(PROTOCOL.read_text())
    contract.update(training_sequences=2, outer_sequences=2, steps=20, updates_per_arm=1)
    row = run_seed(1915, contract)
    assert row["numerical_checks_pass"] is True
    assert row["labels_used_in_learning_or_selection"] is False
    assert list(row["arms"]) == contract["arms"]
    for key, arm in row["arms"].items():
        pmf = np.asarray(arm["pmf"])
        assert pmf.shape == (4, 260)
        assert np.isfinite(pmf).all() and (pmf > 0).all()
        np.testing.assert_allclose(pmf.sum(1), 1, atol=1e-12)
        if key != "em":
            assert arm["frozen_nuisance_preserved"] is True
            assert len(arm["history"]) == 1
            assert arm["public_reference_ll_error"] <= 1e-7
        else:
            assert arm["history"][-1]["completed_updates"] == 1
