# This code builds upon following:
# https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py
# https://github.com/ccrpRepo/mocap_retarget/blob/master/src/mocap/src/robot_ik.py

import logging

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

logger = logging.getLogger(__name__)


class Kinematics:
    def __init__(self, ee_frame) -> None:
        self.frame_name = ee_frame

    def buildFromMJCF(self, mcjf_file):
        self.arm = pin.RobotWrapper.BuildFromMJCF(mcjf_file)
        self.createSolver()

    def buildFromURDF(self, urdf_file):
        self.arm = pin.RobotWrapper.BuildFromURDF(urdf_file)
        self.createSolver()

    def getJac(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.WORLD)
        return J

    def createSolver(self):
        self.model = self.arm.model
        self.data = self.arm.data

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cp_target = casadi.SX.sym("p_target", 3, 1)
        self.caxis_target = casadi.SX.sym("axis_target", 3, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the end-effector frame ID
        self.ee_id = self.model.getFrameId(self.frame_name)

        # Position error: FK translation - target position (3 DOF)
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cp_target],
            [self.cdata.oMf[self.ee_id].translation - self.cp_target],
        )

        # Tool-axis direction error: FK z-axis - target axis (2 effective DOF)
        # The z-column of the EE rotation matrix is the tool approach direction.
        ee_z_axis = self.cdata.oMf[self.ee_id].rotation[:, 2]
        self.axis_error = casadi.Function(
            "axis_error",
            [self.cq, self.caxis_target],
            [ee_z_axis - self.caxis_target],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)
        self.param_p_target = self.opti.parameter(3, 1)
        self.param_axis_target = self.opti.parameter(3, 1)

        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_p_target))
        self.axis_cost = casadi.sumsqr(self.axis_error(self.var_q, self.param_axis_target))
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        # 5DOF task-space projection:
        #   3 position constraints + 2 effective direction constraints = 5 = nq
        #   Rotation around tool axis is left free (unconstrained).
        self.opti.minimize(
            20.0 * self.translational_cost
            + 10.0 * self.axis_cost
            + 0.005 * self.smooth_cost
        )

        ##### IPOPT #####
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 80,
                'tol': 1e-6,
                'acceptable_tol': 1e-4,
                'acceptable_iter': 5,
            },
            'print_time': False,
            'calc_lam_p': False,
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)

    def fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        se3_obj = self.data.oMf[self.ee_id]
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = se3_obj.rotation
        tf[:3, 3] = se3_obj.translation
        return tf

    def ik(self, p_target: np.ndarray, axis_target: np.ndarray,
           current_arm_motor_q: np.ndarray = None):
        """5DOF inverse kinematics with tool-axis direction constraint.

        Args:
            p_target:    (3,) target position [x, y, z] in metres.
            axis_target: (3,) target tool-axis direction (unit vector).
            current_arm_motor_q: (nq,) current joint angles for warm-start
                and smoothness regularisation.

        Returns:
            (dof, info) where *dof* is (nq,) joint solution and
            *info* is a dict with 'success' (bool) and 'sol_tauff'.
        """
        if current_arm_motor_q is not None:
            self.init_data = current_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_p_target, p_target)
        self.opti.set_value(self.param_axis_target, axis_target)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.init_data = sol_q

            v = np.zeros(self.model.nv)
            sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
            sol_tauff = np.concatenate([sol_tauff, np.zeros(self.model.nq - sol_tauff.shape[0])], axis=0)

            info = {"sol_tauff": sol_tauff, "success": True}
            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof, info

        except Exception as e:
            logger.warning("IK convergence failed: %s", e)

            try:
                sol_q = self.opti.debug.value(self.var_q)
            except Exception:
                sol_q = self.init_data.copy()

            self.init_data = sol_q

            info = {"sol_tauff": np.zeros(self.model.nq), "success": False}
            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof, info


if __name__ == "__main__":
    arm = Kinematics("JawOffset")
    arm.buildFromMJCF("../model/trs_so_arm100/so_arm100.xml")

    # Test: reach position (0.1, 0, 0.3) with tool axis pointing down
    p_target = np.array([0.1, 0.0, 0.3])
    axis_target = np.array([0.0, 0.0, -1.0])
    dof, info = arm.ik(p_target, axis_target)
    print(f"DoF: {dof}, Success: {info['success']}")

    tf_result = arm.fk(dof)
    print(f"FK position: {tf_result[:3, 3]}")
    print(f"FK tool axis: {tf_result[:3, 2]}")

    pos_err = np.linalg.norm(tf_result[:3, 3] - p_target)
    axis_err = np.arccos(np.clip(np.dot(tf_result[:3, 2], axis_target), -1, 1))
    print(f"Position error: {pos_err*1000:.2f} mm")
    print(f"Direction error: {np.degrees(axis_err):.2f} deg")
