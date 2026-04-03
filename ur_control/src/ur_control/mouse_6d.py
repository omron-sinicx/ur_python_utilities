import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class Mouse6D():
    """ Subscribe to the 3DConnextion mouse convert messages """
    def __init__(self, twist_topic='/spacenav/twist', joy_topic='/spacenav/joy'):
        self.twist_topic = twist_topic
        self.joy_topic = joy_topic

        self.twist = None
        self.joy_axes = None
        self.joy_buttons = None
        self.last_twist_time = None
        self.last_joy_time = None

        self.twist_sub = rospy.Subscriber(self.twist_topic, Twist, callback=self.twist_cb, queue_size=1)
        self.joy_sub = rospy.Subscriber(self.joy_topic, Joy, callback=self.joy_cb, queue_size=1)

        # Wait for publisher
        rospy.sleep(0.01)

    def has_twist(self):
        return self.last_twist_time is not None

    def twist_age(self):
        if self.last_twist_time is None:
            return None
        return rospy.get_time() - self.last_twist_time

    def twist_is_stale(self, timeout=1.0):
        age = self.twist_age()
        return age is None or age > timeout

    def twist_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{spacenav/twist} topic.
        @type  msg: geometry_msgs/JointState
        @param msg: The Twist message published by the 3DConnextion hardware interface.
        """
        self.last_twist_time = rospy.get_time()
        self.twist = []
        self.twist.append(msg.linear.x)
        self.twist.append(msg.linear.y)
        self.twist.append(msg.linear.z)
        self.twist.append(msg.angular.x)
        self.twist.append(msg.angular.y)
        self.twist.append(msg.angular.z)

    def joy_cb(self, msg):
        """
        Callback executed every time a message is publish in the C{spacenav/joy} topic.
        @type  msg: sensor_msgs/Joy
        @param msg: The Joy message published by the 3DConnextion hardware interface.
        """
        self.last_joy_time = rospy.get_time()
        self.joy_axes = msg.axes
        self.joy_buttons = msg.buttons


def main():
    rospy.init_node("Mouse6D")
    Mouse6D()
    rospy.sleep(1)

if __name__ == '__main__':
    main()
