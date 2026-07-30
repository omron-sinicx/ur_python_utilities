
import time

import rclpy
import controller_manager_msgs.srv
import std_srvs.srv
from ur_control import conversions
from ur_control.utils import solve_namespace, read_parameter
import ur_dashboard_msgs.srv
import ur_dashboard_msgs.msg
import ur_msgs.srv

from std_msgs.msg import Bool


def check_for_real_robot(func):
    '''Decorator that validates the real robot is used or no'''

    def wrap(*args, **kwargs):
        if args[0].use_real_robot:
            return func(*args, **kwargs)
        args[0].node.get_logger().debug("Ignoring function %s since no real robot is being used" % func.__name__)
        return True
    return wrap


class URServices():
    """
    Universal Robots driver specific services.

    Requires a spinning rclpy node (MultiThreadedExecutor in a background thread): the
    synchronous ``client.call`` calls below rely on the executor processing responses.
    """

    #: Seconds to wait for the dashboard client when probing for a real robot.
    SERVICE_DISCOVERY_TIMEOUT = 2.0
    #: Seconds to wait for any service response. rclpy's ``call()`` waits forever when
    #: given no timeout, which turns a missing server into a hung process.
    SERVICE_CALL_TIMEOUT = 5.0

    def __init__(self, node, namespace):
        self.node = node

        self.ns = solve_namespace(namespace, node=node)

        self.ur_ros_control_running_on_robot = False
        self.robot_safety_mode = None
        self.robot_status = dict()

        # Dashboard services live on ur_robot_driver's dashboard_client node, which
        # ur_control.launch.py starts only for a real robot (it is conditioned on NOT
        # use_mock_hardware), never in simulation.
        self.ur_dashboard_clients = {
            "get_loaded_program":     node.create_client(ur_dashboard_msgs.srv.GetLoadedProgram, self.ns + 'dashboard_client/get_loaded_program'),
            "program_running":        node.create_client(ur_dashboard_msgs.srv.IsProgramRunning, self.ns + 'dashboard_client/program_running'),
            "load_program":           node.create_client(ur_dashboard_msgs.srv.Load, self.ns + 'dashboard_client/load_program'),
            "play":                   node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/play'),
            "stop":                   node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/stop'),
            "quit":                   node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/quit'),
            "connect":                node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/connect'),
            "close_popup":            node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/close_popup'),
            "unlock_protective_stop": node.create_client(std_srvs.srv.Trigger, self.ns + 'dashboard_client/unlock_protective_stop'),
            "is_in_remote_control":   node.create_client(ur_dashboard_msgs.srv.IsInRemoteControl, self.ns + 'dashboard_client/is_in_remote_control'),
            "get_program_state":      node.create_client(ur_dashboard_msgs.srv.GetProgramState, self.ns + 'dashboard_client/program_state'),
        }

        # Everything else the driver exposes to us is on the io_and_status_controller
        # (ur_controllers/GPIOController), which replaced ROS 1's ur_hardware_interface.
        self.set_payload_srv = node.create_client(ur_msgs.srv.SetPayload, self.ns + 'io_and_status_controller/set_payload')
        self.speed_slider = node.create_client(ur_msgs.srv.SetSpeedSliderFraction, self.ns + 'io_and_status_controller/set_speed_slider')

        self.set_io = node.create_client(ur_msgs.srv.SetIO, self.ns + 'io_and_status_controller/set_io')

        self.sub_status_ = node.create_subscription(Bool, self.ns + 'io_and_status_controller/robot_program_running', self.ros_control_status_callback, 10)
        self.service_proxy_list = node.create_client(controller_manager_msgs.srv.ListControllers, self.ns + 'controller_manager/list_controllers')
        self.service_proxy_switch = node.create_client(controller_manager_msgs.srv.SwitchController, self.ns + 'controller_manager/switch_controller')

        self.sub_robot_safety_mode = node.create_subscription(ur_dashboard_msgs.msg.SafetyMode, self.ns + 'io_and_status_controller/safety_mode', self.safety_mode_callback, 10)

        # Whether we are driving real hardware. Probed from the dashboard client rather
        # than configured: ROS 2 has no global parameter server, so a launch file cannot
        # set this for a separately started script, and a config value that disagrees with
        # reality fails silently in both directions -- False on hardware turns every method
        # below into a no-op that reports success, while True in simulation used to block
        # forever on the first dashboard call. Set the use_real_robot parameter to override.
        self.use_real_robot = read_parameter(node, "use_real_robot", None)
        if self.use_real_robot is None:
            self.use_real_robot = self.ur_dashboard_clients["get_loaded_program"].wait_for_service(
                timeout_sec=self.SERVICE_DISCOVERY_TIMEOUT)
            node.get_logger().info(
                "use_real_robot not set; probed the dashboard client and found %s"
                % ("a real robot" if self.use_real_robot else "no real robot (simulation)"))

    def _call(self, client, request=None, timeout_sec=None):
        """
        Synchronous service call; builds an empty request of the client's type if none given.

        Raises TimeoutError rather than returning None when the server does not answer, so
        a missing service surfaces as a clear error instead of an AttributeError on the
        response further down.
        """
        if request is None:
            request = client.srv_type.Request()
        timeout_sec = self.SERVICE_CALL_TIMEOUT if timeout_sec is None else timeout_sec
        response = client.call(request, timeout_sec=timeout_sec)
        if response is None:
            raise TimeoutError("No response from %s within %.1fs" % (client.srv_name, timeout_sec))
        return response

    @check_for_real_robot
    def safety_mode_callback(self, msg):
        self.robot_safety_mode = msg.mode

    @check_for_real_robot
    def ros_control_status_callback(self, msg):
        self.ur_ros_control_running_on_robot = msg.data

    @check_for_real_robot
    def is_running_normally(self):
        """
        Returns true if the robot is running (no protective stop, not turned off etc).
        """
        return self.robot_safety_mode == 1 or self.robot_safety_mode == 2  # Normal / Reduced

    @check_for_real_robot
    def is_protective_stopped(self):
        """
        Returns true if the robot is in protective stop.
        """
        return self.robot_safety_mode == 3

    @check_for_real_robot
    def unlock_protective_stop(self):
        service_client = self.ur_dashboard_clients["unlock_protective_stop"]
        request = std_srvs.srv.Trigger.Request()
        start_time = time.time()
        self.node.get_logger().info("Attempting to unlock protective stop of " + self.ns)
        response = None
        while rclpy.ok():
            response = self._call(service_client, request)
            if time.time() - start_time > 20.0:
                self.node.get_logger().error("Timeout of 20s exceeded in unlock protective stop")
                break
            if response.success:
                break
            time.sleep(0.2)
        self._call(self.ur_dashboard_clients["stop"])
        if response is None or not response.success:
            self.node.get_logger().warn("Could not unlock protective stop of " + self.ns + "!")
        return bool(response and response.success)

    @check_for_real_robot
    def set_speed_scale(self, scale):
        try:
            self._call(self.speed_slider,
                       ur_msgs.srv.SetSpeedSliderFraction.Request(speed_slider_fraction=float(scale)))
        except Exception:
            self.node.get_logger().error("Failed to communicate with Dashboard when setting speed slider")
            return False

    @check_for_real_robot
    def set_payload(self, mass, center_of_gravity):
        """
            mass float
            center_of_gravity list[3]
        """
        self.activate_ros_control_on_ur()
        try:
            payload = ur_msgs.srv.SetPayload.Request()
            payload.mass = float(mass)
            payload.center_of_gravity = conversions.to_vector3(center_of_gravity)
            self._call(self.set_payload_srv, payload)
            return True
        except Exception as e:
            self.node.get_logger().error("Exception trying to set payload: %s" % e)
        return False

    def call_service(self, service_name, wait_time=0, retry=True):
        try:
            response = self._call(self.ur_dashboard_clients[service_name])
            self.node.get_logger().debug(f"{service_name} {response=}")
            time.sleep(wait_time)
            return response
        except Exception as e:
            if retry and len(e.args) and "Failed to send request to dashboard server" in str(e.args[0]):
                self.node.get_logger().warn("Call to service failed, retrying connection to dashboard")
                if self.reset_connection():
                    return self.call_service(service_name, retry=False)
            self.node.get_logger().error("Unable to automatically activate robot. Manually activate the robot by pressing 'play' in the polyscope or turn ON the remote control mode.")
            raise e

    @check_for_real_robot
    def wait_for_control_status_to_turn_on(self, wait_time):
        start = time.time()
        while (time.time() - start) < wait_time and rclpy.ok():
            self.node.get_logger().debug(f'{self.ur_ros_control_running_on_robot=}')
            if self.ur_ros_control_running_on_robot:
                response = self.call_service('get_program_state')
                if response.success:
                    if response.state.state == 'PLAYING':
                        return True
                    else:
                        self.call_service('stop')
                        self.call_service('play')
            time.sleep(.1)
        return False

    @check_for_real_robot
    def reset_connection(self):
        try:
            self.node.get_logger().debug("Try to quit before connecting.")
            response = self._call(self.ur_dashboard_clients["quit"])
        except Exception:
            # Ignore failures trying to reset if we cannot communicate with dashboard
            pass

        try:
            self.node.get_logger().debug("Try to connect to dashboard service.")
            response = self._call(self.ur_dashboard_clients["connect"])
            return response.success
        except Exception:
            self.node.get_logger().error("Unable to reset connection...")
            return False

    @check_for_real_robot
    def restart_program(self):
        self.node.get_logger().debug("Try to stop program.")
        response = self.call_service('stop')
        time.sleep(1)
        if response.success:
            self.node.get_logger().debug("Try to play program.")
            response = self.call_service('play')
            time.sleep(1)
        return response.success

    @check_for_real_robot
    def activate_ros_control_on_ur(self, recursion_depth=0):
        # Check if URCap is already running on UR
        if self.wait_for_control_status_to_turn_on(1.0):
            self.node.get_logger().debug("Robot program is running")
            return True
        else:
            self.node.get_logger().info("Robot program not running for " + self.ns)

        try:
            response = self.call_service('is_in_remote_control')
            if not response.success or not response.in_remote_control:
                self.node.get_logger().error(">> Unable to automatically activate robot. Manually activate the robot by pressing 'play' in the polyscope or turn ON the remote control mode.")
                return False
        except Exception:
            pass

        self.node.get_logger().warn(f"Attempt to reconnect # {recursion_depth+1}")

        if recursion_depth > 10:
            self.node.get_logger().error("Tried too often. Breaking out.")
            self.node.get_logger().error("Could not start UR ROS control.")
            raise Exception("Could not activate ROS control on robot " + self.ns + ". Breaking out. Is the UR in Remote Control mode and program installed with correct name?")

        if not rclpy.ok():
            return False

        program_loaded = self.check_loaded_program()

        if not program_loaded:
            self.node.get_logger().warn("Could not load.")
        else:
            # Run the program
            self.node.get_logger().info("Running the program (play)")
            self.restart_program()

        if self.wait_for_control_status_to_turn_on(2.0):
            if self.check_for_dead_controller_and_force_start():
                self.node.get_logger().info("Successfully activated ROS control on robot " + self.ns)
                self.set_speed_scale(scale=1.0)  # Set speed to max always
                return True
        else:
            self.node.get_logger().warn("Failed to start program")
            self.reset_connection()
            return self.activate_ros_control_on_ur(recursion_depth=recursion_depth+1)

    @check_for_real_robot
    def check_loaded_program(self):
        try:
            # Load program if it not loaded already
            response = self._call(self.ur_dashboard_clients["get_loaded_program"])
            if response.program_name == '/programs/ROS_external_control.urp':
                return True
            else:
                self.node.get_logger().info("Currently loaded program was:  " + response.program_name)
                self.node.get_logger().info("Loading ROS control on robot " + self.ns)
                request = ur_dashboard_msgs.srv.Load.Request()
                request.filename = "ROS_external_control.urp"
                response = self._call(self.ur_dashboard_clients["load_program"], request)
                if response.success:  # Try reconnecting to dashboard
                    return True
                else:
                    self.node.get_logger().error("Could not load the ROS_external_control.urp URCap. Is the UR in Remote Control mode and program installed with correct name?")
                for i in range(10):
                    time.sleep(0.2)
                    response = self._call(self.ur_dashboard_clients["get_loaded_program"])
                    if response.program_name == '/programs/ROS_external_control.urp':
                        break
        except Exception:
            self.node.get_logger().warn("Dashboard service did not respond!")
        return False

    @check_for_real_robot
    def check_for_dead_controller_and_force_start(self):
        list_req = controller_manager_msgs.srv.ListControllers.Request()
        switch_req = controller_manager_msgs.srv.SwitchController.Request()
        self.node.get_logger().info("Checking for dead controllers for robot " + self.ns)
        list_res = self._call(self.service_proxy_list, list_req)
        for c in list_res.controller:
            if c.name == "scaled_joint_trajectory_controller":
                if c.state == "inactive":
                    # Force restart
                    self.node.get_logger().warn("Force restart of controller")
                    switch_req.activate_controllers = ['scaled_joint_trajectory_controller']
                    switch_req.strictness = 1
                    switch_res = self._call(self.service_proxy_switch, switch_req)
                    time.sleep(1)
                    return switch_res.ok
                else:
                    self.node.get_logger().info("Controller state is " + c.state + ", returning True.")
                    return True

    @check_for_real_robot
    def load_and_execute_program(self, program_name="", recursion_depth=0, skip_ros_activation=False):
        if not skip_ros_activation:
            self.activate_ros_control_on_ur()
        if not self.load_program(program_name, recursion_depth):
            return False
        return self.execute_loaded_program()

    @check_for_real_robot
    def load_program(self, program_name="", recursion_depth=0):
        if recursion_depth > 10:
            self.node.get_logger().error("Tried too often. Breaking out.")
            self.node.get_logger().error("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
            return False

        load_success = False
        try:
            # Try to stop running program
            self._call(self.ur_dashboard_clients["stop"])
            time.sleep(.5)

            # Load program if it not loaded already
            response = self._call(self.ur_dashboard_clients["get_loaded_program"])
            if response.program_name == '/programs/' + program_name:
                return True
            else:
                self.node.get_logger().info("Loaded program is different %s. Attempting to load new program %s" % (response.program_name, program_name))
                request = ur_dashboard_msgs.srv.Load.Request()
                request.filename = program_name
                response = self._call(self.ur_dashboard_clients["load_program"], request)
                if response.success:  # Try reconnecting to dashboard
                    load_success = True
                    return True
                else:
                    self.node.get_logger().error("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
        except Exception:
            self.node.get_logger().warn("Dashboard service did not respond to load_program!")
        if not load_success:
            self.node.get_logger().warn("Waiting and trying again")
            time.sleep(3)
            try:
                if recursion_depth > 0:  # If connect alone failed, try quit and then connect
                    response = self._call(self.ur_dashboard_clients["quit"])
                    self.node.get_logger().error("Program could not be loaded on UR: " + program_name)
                    time.sleep(.5)
            except Exception:
                self.node.get_logger().warn("Dashboard service did not respond to quit! ")
                pass
            response = self._call(self.ur_dashboard_clients["connect"])
            time.sleep(.5)
            return self.load_program(program_name=program_name, recursion_depth=recursion_depth+1)

    @check_for_real_robot
    def execute_loaded_program(self):
        # Run the program
        try:
            response = self._call(self.ur_dashboard_clients["play"])
            if not response.success:
                self.node.get_logger().error("Could not start program. Is the UR in Remote Control mode and program installed with correct name?")
                return False
            else:
                self.node.get_logger().info("Successfully started program on robot " + self.ns)
                return True
        except Exception as e:
            self.node.get_logger().error(str(e))
            return False

    @check_for_real_robot
    def close_ur_popup(self):
        # Close a popup on the teach pendant to continue program execution
        response = self._call(self.ur_dashboard_clients["close_popup"])
        if not response.success:
            self.node.get_logger().error("Could not close popup.")
            return False
        else:
            self.node.get_logger().info("Successfully closed popup on teach pendant of robot " + self.ns)
            return True
