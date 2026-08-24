from __future__ import annotations

import time
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from mc_one.action import ComeToMe, GoToPlace, PlayAnimation, SimpleMove
from mc_one.msg import Localization3D, PoseDetection, PoseDetections


class FakeSkillServers(Node):
    """Local-only fake action servers for AI-BT ROS smoke tests.

    This is deliberately not a simulator. It returns successful action results
    with the same action names as navigation/animator and publishes simple
    perception facts so the AI-BT stack can be exercised without Isaac.
    """

    def __init__(self) -> None:
        super().__init__("mc_ai_bt_fake_skill_servers")
        self.declare_parameter("delay_sec", 0.05)
        self.declare_parameter("fake_object_name", "cup")
        self.declare_parameter("fake_object_score", 0.9)
        self.declare_parameter("fake_visible_people_count", 1)
        self.declare_parameter("perception_period_sec", 0.5)

        self._object_pub = self.create_publisher(
            Localization3D,
            "/mc_perception/object_localization",
            10,
        )
        self._people_pub = self.create_publisher(
            PoseDetections,
            "/mc_perception/pose_detections",
            10,
        )
        period = max(0.1, float(self.get_parameter("perception_period_sec").value or 0.5))
        self.create_timer(period, self._publish_fake_perception)

        self._go_to_place = ActionServer(
            self,
            GoToPlace,
            "/mc_navigation/go_to_place",
            self._execute_go_to_place,
        )
        self._come_to_me = ActionServer(
            self,
            ComeToMe,
            "/mc_navigation/come_to_me",
            self._execute_come_to_me,
        )
        self._simple_move = ActionServer(
            self,
            SimpleMove,
            "/mc_navigation/simple_move",
            self._execute_simple_move,
        )
        self._play_animation = ActionServer(
            self,
            PlayAnimation,
            "/mc_animator/play",
            self._execute_play_animation,
        )

    def _execute_go_to_place(self, goal_handle):
        goal = goal_handle.request
        result = GoToPlace.Result()
        if self._maybe_cancel(goal_handle, result, "go_to_place canceled"):
            return result
        feedback = GoToPlace.Feedback()
        feedback.distance_remaining = 0.0
        feedback.navigation_time = self._delay_sec()
        feedback.number_of_recoveries = 0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result.success = True
        result.message = f"fake arrived at {goal.name}"
        return result

    def _execute_come_to_me(self, goal_handle):
        result = ComeToMe.Result()
        if self._maybe_cancel(goal_handle, result, "come_to_me canceled"):
            return result
        feedback = ComeToMe.Feedback()
        feedback.distance_remaining = 0.0
        feedback.navigation_time = self._delay_sec()
        feedback.number_of_recoveries = 0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result.success = True
        result.message = "fake arrived near interaction owner"
        return result

    def _execute_simple_move(self, goal_handle):
        goal = goal_handle.request
        result = SimpleMove.Result()
        if self._maybe_cancel(goal_handle, result, "simple_move canceled"):
            return result
        feedback = SimpleMove.Feedback()
        feedback.traveled = float(goal.value)
        feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result.success = True
        result.message = f"fake simple_move {goal.action} {goal.value:g}"
        return result

    def _execute_play_animation(self, goal_handle):
        goal = goal_handle.request
        result = PlayAnimation.Result()
        if self._maybe_cancel(goal_handle, result, "play_animation canceled"):
            return result
        feedback = PlayAnimation.Feedback()
        feedback.progress = 1.0
        feedback.clip = goal.animation
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result.success = True
        result.message = f"fake animation played: {goal.animation}"
        return result

    def _maybe_cancel(self, goal_handle, result: Any, message: str) -> bool:
        delay = self._delay_sec()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = message
            return True
        time.sleep(delay)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = message
            return True
        return False

    def _publish_fake_perception(self) -> None:
        object_name = str(self.get_parameter("fake_object_name").value or "").strip()
        if object_name:
            object_msg = Localization3D()
            object_msg.object_name = object_name
            object_msg.x = 1.0
            object_msg.y = 0.0
            object_msg.z = 0.5
            object_msg.score = float(self.get_parameter("fake_object_score").value or 0.9)
            self._object_pub.publish(object_msg)

        count = max(0, int(self.get_parameter("fake_visible_people_count").value or 0))
        people_msg = PoseDetections()
        people_msg.header.stamp = self.get_clock().now().to_msg()
        people_msg.poses = [_fake_pose(idx) for idx in range(count)]
        self._people_pub.publish(people_msg)

    def _delay_sec(self) -> float:
        return max(0.0, float(self.get_parameter("delay_sec").value or 0.05))


def _fake_pose(idx: int) -> PoseDetection:
    pose = PoseDetection()
    pose.nose_px = Point(x=320.0 + idx * 10.0, y=180.0, z=0.0)
    pose.face_size_px = 80.0
    pose.landmarks = []
    return pose


def main() -> None:
    rclpy.init()
    node = FakeSkillServers()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        _safe_rclpy_shutdown()


def _safe_rclpy_shutdown() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
