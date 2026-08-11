#!/usr/bin/env python3
"""Test suite for 5DOF tool-axis IK solver.

Run from project root:
    python tests/test_ik_tool_axis.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.src.pinocchio_kinematic import Kinematics

ARM_XML_PATH = str(PROJECT_ROOT / "common" / "model" / "trs_so_arm100" / "so_arm100.xml")


def build_solver():
    arm = Kinematics("Jaw")
    arm.buildFromMJCF(ARM_XML_PATH)
    return arm


def test_fk_roundtrip(arm: Kinematics):
    """Test 1: FK -> IK -> FK roundtrip at multiple known joint configurations."""
    print("\n=== Test 1: FK -> IK -> FK roundtrip ===")

    test_configs = [
        np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0]),          # near home
        np.array([0.3, -0.8, 1.0, -0.2, -1.0, 0.0]),         # bent configuration
        np.array([-0.5, -0.3, 0.3, 0.1, 0.5, 0.0]),          # opposite side
        np.array([0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0]),  # default guess_q
    ]

    all_pass = True
    for i, q_true in enumerate(test_configs):
        q_input = q_true[:arm.model.nq]

        # FK to get ground truth
        tf = arm.fk(q_input)
        p_true = tf[:3, 3]
        axis_true = tf[:3, 2]  # z-column = tool axis

        # IK to recover joints
        dof, info = arm.ik(p_true, axis_true, current_arm_motor_q=q_input)
        success = info.get("success", False)

        # FK again to verify
        tf_recovered = arm.fk(dof[:arm.model.nq])
        p_recovered = tf_recovered[:3, 3]
        axis_recovered = tf_recovered[:3, 2]

        pos_err_mm = np.linalg.norm(p_recovered - p_true) * 1000
        axis_angle_deg = np.degrees(np.arccos(np.clip(np.dot(axis_recovered, axis_true), -1, 1)))

        status = "PASS" if (success and pos_err_mm < 1.0 and axis_angle_deg < 2.0) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  [{status}] config {i}: "
            f"ik_success={success}, "
            f"pos_err={pos_err_mm:.3f}mm, "
            f"axis_err={axis_angle_deg:.3f}deg"
        )

    return all_pass


def test_various_directions(arm: Kinematics):
    """Test 2: IK with various tool axis directions from a reachable position."""
    print("\n=== Test 2: Various tool axis directions ===")

    # Start from a known good config to get a reachable position
    q_home = np.array([0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0])
    tf_home = arm.fk(q_home[:arm.model.nq])
    p_base = tf_home[:3, 3]

    test_axes = [
        ("down",       np.array([0.0, 0.0, -1.0])),
        ("up",         np.array([0.0, 0.0, 1.0])),
        ("forward",    np.array([0.0, -1.0, 0.0])),
        ("tilt 45",    np.array([0.0, -0.707, -0.707])),
        ("tilt -45",   np.array([0.0, 0.707, -0.707])),
        ("side tilt",  np.array([0.5, -0.5, -0.707])),
    ]

    all_pass = True
    for name, axis_target in test_axes:
        axis_target = axis_target / np.linalg.norm(axis_target)
        dof, info = arm.ik(p_base, axis_target, current_arm_motor_q=q_home[:arm.model.nq])
        success = info.get("success", False)

        if success:
            tf_result = arm.fk(dof[:arm.model.nq])
            p_result = tf_result[:3, 3]
            axis_result = tf_result[:3, 2]
            pos_err_mm = np.linalg.norm(p_result - p_base) * 1000
            axis_angle_deg = np.degrees(np.arccos(np.clip(np.dot(axis_result, axis_target), -1, 1)))
            status = "PASS" if (pos_err_mm < 5.0 and axis_angle_deg < 5.0) else "WARN"
        else:
            pos_err_mm = float("inf")
            axis_angle_deg = float("inf")
            status = "FAIL"

        if status != "PASS":
            all_pass = False

        print(
            f"  [{status}] {name:12s}: "
            f"ik_success={success}, "
            f"pos_err={pos_err_mm:.2f}mm, "
            f"axis_err={axis_angle_deg:.2f}deg"
        )

    return all_pass


def test_ik_failure_graceful(arm: Kinematics):
    """Test 3: IK should return success=False for unreachable targets (not crash)."""
    print("\n=== Test 3: Graceful failure for unreachable targets ===")

    unreachable_targets = [
        ("too far",   np.array([1.0, 0.0, 0.0]),    np.array([0, 0, -1])),
        ("behind",    np.array([0.0, 0.5, 0.0]),     np.array([0, 0, -1])),
        ("too high",  np.array([0.0, 0.0, 2.0]),     np.array([0, 0, -1])),
    ]

    all_pass = True
    q_init = np.zeros(arm.model.nq)

    for name, p, axis in unreachable_targets:
        try:
            dof, info = arm.ik(p, axis, current_arm_motor_q=q_init)
            success = info.get("success", False)
            crashed = False
        except Exception as e:
            crashed = True
            success = False

        # We expect failure, NOT crash
        status = "PASS" if (not crashed and not success) else ("CRASH" if crashed else "WARN")
        if status != "PASS":
            all_pass = False
            if status == "WARN":
                # IK claimed success for unreachable target — check actual error
                tf = arm.fk(dof[:arm.model.nq])
                pos_err = np.linalg.norm(tf[:3, 3] - p) * 1000
                print(f"  [{status}] {name:10s}: IK claimed success but pos_err={pos_err:.1f}mm")
                continue

        print(f"  [{status}] {name:10s}: crashed={crashed}, ik_success={success}")

    return all_pass


def test_axis_is_unit_vector(arm: Kinematics):
    """Test 4: Verify FK tool axis is always unit length."""
    print("\n=== Test 4: FK tool axis unit vector ===")

    rng = np.random.default_rng(42)
    q_lower = arm.model.lowerPositionLimit
    q_upper = arm.model.upperPositionLimit

    all_pass = True
    for i in range(20):
        q_rand = q_lower + rng.random(arm.model.nq) * (q_upper - q_lower)
        tf = arm.fk(q_rand)
        axis = tf[:3, 2]
        norm = np.linalg.norm(axis)
        err = abs(norm - 1.0)
        if err > 1e-6:
            print(f"  [FAIL] sample {i}: ||axis||={norm:.9f}")
            all_pass = False

    if all_pass:
        print(f"  [PASS] all 20 random configs: ||axis|| = 1.0 (err < 1e-6)")
    return all_pass


def test_weight_effectiveness(arm: Kinematics):
    """Test 5: Verify direction weight is effective (axis_cost weight = 10.0 should give good direction accuracy)."""
    print("\n=== Test 5: Direction weight effectiveness ===")

    q_home = np.array([0.0547, -0.4193, 0.6309, -0.2114, -1.5357, 0.0])
    tf_home = arm.fk(q_home[:arm.model.nq])
    p_target = tf_home[:3, 3]

    # Target a direction different from current
    current_axis = tf_home[:3, 2]
    # Rotate 30 degrees around Y axis
    angle = np.radians(30)
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    target_axis = Ry @ current_axis
    target_axis = target_axis / np.linalg.norm(target_axis)

    dof, info = arm.ik(p_target, target_axis, current_arm_motor_q=q_home[:arm.model.nq])
    success = info.get("success", False)

    all_pass = True
    if success:
        tf_result = arm.fk(dof[:arm.model.nq])
        axis_result = tf_result[:3, 2]
        axis_err_deg = np.degrees(np.arccos(np.clip(np.dot(axis_result, target_axis), -1, 1)))
        pos_err_mm = np.linalg.norm(tf_result[:3, 3] - p_target) * 1000

        # With weight=10.0, direction should be accurate (<5 deg)
        status = "PASS" if axis_err_deg < 5.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  [{status}] 30deg rotation test: "
            f"pos_err={pos_err_mm:.2f}mm, axis_err={axis_err_deg:.2f}deg"
        )
    else:
        print(f"  [FAIL] IK failed for 30deg rotation test")
        all_pass = False

    return all_pass


def main():
    print("Building IK solver from MJCF model...")
    arm = build_solver()
    print(f"Model: nq={arm.model.nq}, frame='{arm.frame_name}'")

    results = {}
    results["FK roundtrip"] = test_fk_roundtrip(arm)
    results["Various directions"] = test_various_directions(arm)
    results["Graceful failure"] = test_ik_failure_graceful(arm)
    results["Unit vector"] = test_axis_is_unit_vector(arm)
    results["Weight effectiveness"] = test_weight_effectiveness(arm)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {name}")

    print("=" * 50)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 50)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
