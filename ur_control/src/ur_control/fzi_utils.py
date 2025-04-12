# The MIT License (MIT)
#
# Copyright (c) 2023 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import numpy as np
import rospy
from ur_control.constants import JOINT_TRAJECTORY_CONTROLLER, CARTESIAN_COMPLIANCE_CONTROLLER, ExecutionResult


def is_more_extreme(value, target):
    """
    Check if a value is more extreme than a target value.

    For each dimension:
    - If target is positive, check if value is greater than target
    - If target is negative, check if value is less than target

    Args:
        value (numpy.ndarray): The value to check
        target (numpy.ndarray): The target value to compare against

    Returns:
        bool: True if the value is more extreme than the target, False otherwise
    """
    # Check each dimension individually
    for i in range(len(target)):
        # For positive targets, check if value exceeds target
        if target[i] > 0 and value[i] > target[i]:
            return True
        # For negative targets, check if value is more negative than target
        elif target[i] < 0 and value[i] < target[i]:
            return True

    # If we get here, no dimension exceeded its target
    return False


def convert_selection_matrix_to_parameters(selection_matrix):
    """
    Convert a selection matrix to controller parameters.

    Args:
        selection_matrix (numpy.ndarray): A 6-element array representing the selection matrix

    Returns:
        dict: A dictionary containing the selection matrix parameters
    """
    return {
        "stiffness":
        {
            "sel_x": selection_matrix[0],
            "sel_y": selection_matrix[1],
            "sel_z": selection_matrix[2],
            "sel_ax": selection_matrix[3],
            "sel_ay": selection_matrix[4],
            "sel_az": selection_matrix[5],
        }
    }


def convert_stiffness_to_parameters(stiffness):
    """
    Convert stiffness values to controller parameters.

    Args:
        stiffness (numpy.ndarray): A 6-element array representing the stiffness values

    Returns:
        dict: A dictionary containing the stiffness parameters
    """
    return {
        "stiffness":
        {
            "trans_x": stiffness[0],
            "trans_y": stiffness[1],
            "trans_z": stiffness[2],
            "rot_x": stiffness[3],
            "rot_y": stiffness[4],
            "rot_z": stiffness[5],
        }
    }


def convert_pd_gains_to_parameters(p_gains, d_gains=[0, 0, 0, 0, 0, 0]):
    """
    Convert P and D gains to controller parameters.

    Args:
        p_gains (numpy.ndarray): A 6-element array representing the P gains
        d_gains (numpy.ndarray, optional): A 6-element array representing the D gains. Defaults to zeros.

    Returns:
        dict: A dictionary containing the P and D gain parameters
    """
    return {
        "trans_x": {"p": p_gains[0], "d": d_gains[0]},
        "trans_y": {"p": p_gains[1], "d": d_gains[1]},
        "trans_z": {"p": p_gains[2], "d": d_gains[2]},
        "rot_x": {"p": p_gains[3], "d": d_gains[3]},
        "rot_y": {"p": p_gains[4], "d": d_gains[4]},
        "rot_z": {"p": p_gains[5], "d": d_gains[5]}
    }


def switch_cartesian_controllers(func):
    """
    Decorator that switches from cartesian to joint trajectory controllers and back.

    This decorator ensures that the cartesian controller is activated before the function is called
    and that the joint trajectory controller is activated after the function returns.

    Args:
        func (callable): The function to decorate

    Returns:
        callable: The decorated function
    """
    def wrap(*args, **kwargs):
        if not args[0].auto_switch_controllers:
            return func(*args, **kwargs)

        args[0].activate_cartesian_controller()

        try:
            res = func(*args, **kwargs)
        except Exception as e:
            rospy.logerr("Exception: %s" % e)
            res = ExecutionResult.DONE

        args[0].activate_joint_trajectory_controller()

        return res
    return wrap
