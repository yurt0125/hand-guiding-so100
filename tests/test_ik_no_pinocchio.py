#!/usr/bin/env python3
"""Test IK structure and utility functions without pinocchio/mujoco.

Only requires: numpy, casadi
Run: python tests/test_ik_no_pinocchio.py
"""

import math
import sys
from dataclasses import dataclass
from typing import Optional

import casadi
import numpy as np


# ---- Inline the functions we need (avoid mujoco/pinocchio imports) ----

@dataclass
class EndEffectorTarget:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tool_axis_x: float = 0.0
    tool_axis_y: float = 0.0
    tool_axis_z: float = -1.0
    gripper: float = 0.0
    valid: bool = True
    source: str = "unknown"
    timestamp: Optional[float] = None

    def tool_axis(self) -> np.ndarray:
        a = np.array([self.tool_axis_x, self.tool_axis_y, self.tool_axis_z])
        norm = np.linalg.norm(a)
        if norm < 1e-9:
            return np.array([0.0, 0.0, -1.0])
        return a / norm


def _normalize_axis(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


def _slerp_axis(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    angle = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    if angle < 1e-9:
        return a.copy()
    if angle <= max_angle:
        return b.copy()
    t = max_angle / angle
    sin_angle = np.sin(angle)
    if sin_angle < 1e-9:
        return a.copy()
    result = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / sin_angle
    return _normalize_axis(result)


# ---- Tests ----

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def test_end_effector_target():
    print("\n=== Test 1: EndEffectorTarget.tool_axis() ===")

    t = EndEffectorTarget()
    a = t.tool_axis()
    check("default = [0,0,-1]", np.allclose(a, [0, 0, -1]))
    check("default is unit", abs(np.linalg.norm(a) - 1.0) < 1e-9)

    t2 = EndEffectorTarget(tool_axis_x=0, tool_axis_y=0, tool_axis_z=0)
    check("zero fallback = [0,0,-1]", np.allclose(t2.tool_axis(), [0, 0, -1]))

    t3 = EndEffectorTarget(tool_axis_x=3, tool_axis_y=4, tool_axis_z=0)
    a3 = t3.tool_axis()
    check("normalize (3,4,0) -> (0.6,0.8,0)", np.allclose(a3, [0.6, 0.8, 0.0], atol=1e-6))
    check("normalized is unit", abs(np.linalg.norm(a3) - 1.0) < 1e-9)


def test_axis_utilities():
    print("\n=== Test 2: _normalize_axis, _slerp_axis ===")

    check("normalize zero -> default", np.allclose(_normalize_axis(np.zeros(3)), [0, 0, -1]))
    check("normalize -> unit", abs(np.linalg.norm(_normalize_axis(np.array([1, 1, 0]))) - 1.0) < 1e-9)

    # slerp: same direction
    check("slerp same dir", np.allclose(_slerp_axis(np.array([0, 0, -1.0]), np.array([0, 0, -1.0]), 0.1), [0, 0, -1]))

    # slerp: 10 deg step on a 90 deg arc
    a_from = np.array([0, 0, -1.0])
    a_to = np.array([0, -1, 0.0])
    a_step = _slerp_axis(a_from, a_to, np.radians(10))
    angle = np.degrees(np.arccos(np.clip(np.dot(a_step, a_from), -1, 1)))
    check(f"slerp 10deg step (got {angle:.2f})", 9 < angle < 11)
    check("slerp step is unit", abs(np.linalg.norm(a_step) - 1.0) < 1e-9)

    # slerp: snap when within max
    a_snap = _slerp_axis(a_from, a_to, np.radians(100))
    check("slerp snaps when within max", np.allclose(a_snap, a_to))

    # slerp: 180 degree edge case
    a_opp = _slerp_axis(np.array([0, 0, -1.0]), np.array([0, 0, 1.0]), np.radians(10))
    check("slerp 180deg no NaN", np.all(np.isfinite(a_opp)))
    check("slerp 180deg unit", abs(np.linalg.norm(a_opp) - 1.0) < 1e-6)


def test_keyboard_rodrigues():
    print("\n=== Test 3: Keyboard Rodrigues rotation ===")

    axis = np.array([0.0, 0.0, -1.0])
    s = 0.02

    # I key: rotate around world Y
    delta_I = np.cross([0, 1, 0], axis) * s
    new_I = _normalize_axis(axis + delta_I)
    check("I key tilts axis", not np.allclose(new_I, axis))
    check("I key stays unit", abs(np.linalg.norm(new_I) - 1.0) < 1e-9)

    # J key: rotate around world X
    delta_J = -np.cross([1, 0, 0], axis) * s
    new_J = _normalize_axis(axis + delta_J)
    check("J key tilts axis", not np.allclose(new_J, axis))
    check("J key stays unit", abs(np.linalg.norm(new_J) - 1.0) < 1e-9)

    # 100 consecutive I presses should not cause NaN or denormalization
    a = np.array([0.0, 0.0, -1.0])
    for _ in range(100):
        delta = np.cross([0, 1, 0], a) * s
        a = _normalize_axis(a + delta)
    check("100x I key: no NaN", np.all(np.isfinite(a)))
    check("100x I key: still unit", abs(np.linalg.norm(a) - 1.0) < 1e-9)
    total_angle = np.degrees(np.arccos(np.clip(np.dot(a, [0, 0, -1]), -1, 1)))
    print(f"        (axis drifted {total_angle:.1f} deg from start)")


def test_casadi_2d_axis_cost():
    print("\n=== Test 4: Casadi 2D axis cost ===")

    q = casadi.SX.sym("q")
    a_actual = casadi.vertcat(casadi.cos(q), casadi.sin(q))
    a_target = casadi.SX.sym("at", 2)
    cost = casadi.sumsqr(a_actual - a_target)
    cost_fn = casadi.Function("c", [q, a_target], [cost])

    opti = casadi.Opti()
    vq = opti.variable()
    pa = opti.parameter(2)
    opti.minimize(cost_fn(vq, pa))
    opti.solver("ipopt", {"ipopt": {"print_level": 0}, "print_time": False})

    tests = [
        ("target [0,1]",  [0, 1],  np.pi / 2),
        ("target [-1,0]", [-1, 0], np.pi),
        ("target [1,0]",  [1, 0],  0.0),
    ]
    for name, tgt, expected_q in tests:
        opti.set_value(pa, tgt)
        opti.set_initial(vq, expected_q + 0.3)  # slightly off
        sol = opti.solve()
        q_sol = float(sol.value(vq))
        a_sol = np.array([np.cos(q_sol), np.sin(q_sol)])
        err = np.linalg.norm(a_sol - np.array(tgt))
        check(f"{name}: err={err:.6f}", err < 0.01)


def test_casadi_3d_axis_cost():
    print("\n=== Test 5: Casadi 3D axis cost (2DOF on sphere) ===")

    # Model: tool_axis = Ry(q1) @ Rx(q2) @ [0,0,-1]
    q1 = casadi.SX.sym("q1")
    q2 = casadi.SX.sym("q2")
    c1, s1 = casadi.cos(q1), casadi.sin(q1)
    c2, s2 = casadi.cos(q2), casadi.sin(q2)
    Ry = casadi.vertcat(
        casadi.horzcat(c1, 0, s1),
        casadi.horzcat(0, 1, 0),
        casadi.horzcat(-s1, 0, c1),
    )
    Rx = casadi.vertcat(
        casadi.horzcat(1, 0, 0),
        casadi.horzcat(0, c2, -s2),
        casadi.horzcat(0, s2, c2),
    )
    tool = Ry @ Rx @ casadi.vertcat(0, 0, -1)
    at = casadi.SX.sym("at", 3)
    cost = casadi.sumsqr(tool - at)
    cost_fn = casadi.Function("c3", [q1, q2, at], [cost])

    opti = casadi.Opti()
    vq1 = opti.variable()
    vq2 = opti.variable()
    pa = opti.parameter(3)
    opti.minimize(cost_fn(vq1, vq2, pa))
    opti.subject_to(opti.bounded(-np.pi, vq1, np.pi))
    opti.subject_to(opti.bounded(-np.pi, vq2, np.pi))
    opti.solver("ipopt", {"ipopt": {"print_level": 0}, "print_time": False})

    targets = [
        ("down",    [0, 0, -1],  0.0, 0.0),
        ("forward", [0, -1, 0],  0.0, 0.0),
        ("tilt45",  [0, -0.707, -0.707], 0.0, 0.0),
        ("side",    [0.5, -0.5, -0.707], 0.0, 0.0),
        ("up",      [0, 0, 1],   np.pi, 0.0),  # need good init for antipodal target
    ]
    for name, tgt, init_q1, init_q2 in targets:
        tgt = np.array(tgt, dtype=float)
        tgt = tgt / np.linalg.norm(tgt)
        opti.set_value(pa, tgt)
        opti.set_initial(vq1, init_q1)
        opti.set_initial(vq2, init_q2)
        sol = opti.solve()
        q1v = float(sol.value(vq1))
        q2v = float(sol.value(vq2))
        # Verify FK
        c1v, s1v = np.cos(q1v), np.sin(q1v)
        c2v, s2v = np.cos(q2v), np.sin(q2v)
        Ryv = np.array([[c1v, 0, s1v], [0, 1, 0], [-s1v, 0, c1v]])
        Rxv = np.array([[1, 0, 0], [0, c2v, -s2v], [0, s2v, c2v]])
        a_solved = Ryv @ Rxv @ np.array([0, 0, -1])
        dir_err = np.degrees(np.arccos(np.clip(np.dot(a_solved, tgt), -1, 1)))
        check(f"{name:10s}: axis_err={dir_err:.4f}deg", dir_err < 1.0)


def main():
    test_end_effector_target()
    test_axis_utilities()
    test_keyboard_rodrigues()
    test_casadi_2d_axis_cost()
    test_casadi_3d_axis_cost()

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"FAILED: {failed}/{passed + failed}")
    print("=" * 50)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
