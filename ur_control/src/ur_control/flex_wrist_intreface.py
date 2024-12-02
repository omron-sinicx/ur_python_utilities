import actionlib
import actionlib_tutorials.msg
from o2ac_msgs.msg import *
import rospy

def control_wrist(turn_rigid_on = False, turn_flex_on = False):
    client = actionlib.SimpleActionClient('flex_wrist_control', FlexWristControlAction)
    client.wait_for_server()
    goal = FlexWristControlGoal()

    goal.name = 'flex_wrist_control'
    goal.turn_rigid_on = turn_rigid_on
    goal.turn_flex_on = turn_flex_on

    client.send_goal_and_wait(goal,rospy.Duration(30), rospy.Duration(10))
    client.wait_for_result()

    return client.get_result()
