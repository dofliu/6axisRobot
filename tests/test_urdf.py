import pybullet as pb

from sixaxis.pbsim import URDF_PATH
from sixaxis.urdf import build_urdf


def test_committed_urdf_up_to_date(robot):
    assert URDF_PATH.read_text(encoding="utf-8") == build_urdf(robot)


def test_urdf_mass_inertia_loaded(robot, pbr):
    for i in range(6):
        info = pb.getDynamicsInfo(pbr.id, i, physicsClientId=pbr.cid)
        assert abs(info[0] - robot.m[i]) < 1e-12
        diag = sorted(info[2])
        assert all(abs(a - b) < 1e-9 for a, b in zip(diag, sorted(robot.I_diag[i])))
        assert all(abs(a - b) < 1e-9 for a, b in zip(info[3], robot.com[i]))
