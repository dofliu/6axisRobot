"""T102 取放測試：程式結構、料盤格放置、取點偏差容忍度與重取恢復。"""

import pytest

from sixaxis import scene as S
from sixaxis.commands import Executor, parse
from sixaxis.graspsim import GraspScene
from sixaxis.tasks import Z_POCKET, pick_place_program, run_pick_place

PC = S.pocket_centers()
BASE = (0.40, 0.0)


def _run(true, place, **kw):
    s = GraspScene(workcell=True)
    try:
        return run_pick_place(s, Executor(s), true, place, **kw)
    finally:
        s.close()


def test_program_places_at_pocket_height():
    prog = parse(pick_place_program((0.4, 0.0), PC[2], speed=0.8))
    assert [c["cmd"] for c in prog][4:] == ["move_j", "move_l", "release", "move_l"]
    assert prog[5]["tcp"][2] == pytest.approx(Z_POCKET) and prog[4]["tcp"][:2] == [PC[2][0], PC[2][1]]


@pytest.mark.parametrize("k", [0, 3])
def test_nominal_place_in_pocket(k):
    m = _run((0.38, 0.01), PC[k])
    assert m["success"], m["error"]
    assert m["xy_err_mm"] < 1.5 and abs(m["z_err_mm"]) < 1.0 and m["tilt_deg"] < 1.0
    assert m["torque_util"] < 1.0


def test_y_offset_self_centres_but_x_offset_breaks_raw_pick():
    ok = _run(BASE, PC[1], cmd_pick=(BASE[0], BASE[1] + 0.010))  # 兩指對中，y 偏差被吸收
    bad = _run(BASE, PC[1], cmd_pick=(BASE[0] + 0.012, BASE[1]))  # x 偏差殘留 → 撞料盤側銷
    assert ok["success"] and not bad["success"]


def test_regrasp_recovers_x_offset():
    m = _run(BASE, PC[1], cmd_pick=(BASE[0] + 0.012, BASE[1]), recover=True)
    assert m["success"] and m["attempts"] == 2
    assert m["log"][0]["error"] == "INHAND_OFFSET" and abs(m["log"][0]["dx_mm"] + 12.0) < 0.5


def test_gross_y_error_is_unrecoverable():
    m = _run(BASE, PC[1], cmd_pick=(BASE[0], BASE[1] + 0.018), recover=True)
    assert not m["success"] and m["error"] == "UNRECOVERABLE"
