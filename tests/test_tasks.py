"""T101 點到點搬運測試：程式產生器、取放區、物理搬運成功與精度。"""

import numpy as np
import pytest

from sixaxis.commands import Executor, parse
from sixaxis.graspsim import GraspScene
from sixaxis.tasks import MIN_SEP, Z_OBJ, ZONE_X, ZONE_Y, random_pairs, run_transfer, transfer_program


def test_program_structure():
    prog = parse(transfer_program((0.38, 0.0), (0.42, 0.01), speed=0.8, profile="trapezoid"))
    names = ["move_j", "move_l", "grip", "move_l", "move_j", "move_l", "release", "move_l"]
    assert [c["cmd"] for c in prog] == names
    assert prog[1]["tcp"][2] == pytest.approx(Z_OBJ) and prog[4]["tcp"][:2] == [0.42, 0.01]
    assert prog[0]["speed"] == 0.8 and prog[0]["profile"] == "trapezoid"


def test_random_pairs_reproducible_and_in_zone():
    a, b = random_pairs(20, 5), random_pairs(20, 5)
    assert a == b and a != random_pairs(20, 6)
    for s, d in a:
        for p in (s, d):
            assert ZONE_X[0] <= p[0] <= ZONE_X[1] and ZONE_Y[0] <= p[1] <= ZONE_Y[1]
        assert np.hypot(s[0] - d[0], s[1] - d[1]) >= MIN_SEP


def test_dry_run_all_pairs_reachable():
    for s, d in random_pairs(10, 3):
        rep = Executor().run(transfer_program(s, d))
        assert rep.ok and rep.t_total > 3.0


@pytest.mark.parametrize("profile,force", [("quintic", 40.0), ("trapezoid", 20.0)])
def test_physical_transfer(profile, force):
    s = GraspScene(workcell=True)
    try:
        src, dst = random_pairs(1, 11)[0]
        m = run_transfer(s, Executor(s), src, dst, profile=profile, force=force)
    finally:
        s.close()
    assert m["success"], m["error"]
    assert m["xy_err_mm"] < 2.0 and abs(m["z_err_mm"]) < 1.0 and m["tilt_deg"] < 2.0
    assert m["vel_util"] <= 0.5 + 1e-6 and m["torque_util"] < 1.0


def test_unreachable_destination_reports_error():
    s = GraspScene(workcell=True)
    try:
        m = run_transfer(s, Executor(s), (0.4, 0.0), (2.0, 0.0))
    finally:
        s.close()
    assert not m["success"] and m["error"].startswith("UNREACHABLE")
