from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner_backend",
                default_value="bootstrap",
                description="Planner backend: bootstrap or optional langchain_openai.",
            ),
            DeclareLaunchArgument(
                "planner_model_name",
                default_value="",
                description="Model name for optional LLM planner backends.",
            ),
            DeclareLaunchArgument(
                "planner_temperature",
                default_value="0.0",
                description="Planner sampling temperature for optional LLM backends.",
            ),
            DeclareLaunchArgument(
                "planner_timeout",
                default_value="15.0",
                description="Planner timeout in seconds.",
            ),
            DeclareLaunchArgument(
                "mission_journal_path",
                default_value="",
                description="Optional JSONL path for mission event snapshots.",
            ),
            DeclareLaunchArgument(
                "start_embodied_skills",
                default_value="true",
                description="Start the closed-loop embodied skill action server from seattle_lab.",
            ),
            DeclareLaunchArgument(
                "embodied_allow_fake_success",
                default_value="false",
                description="Allow mc_embodied_skills to return smoke-test success without a real closed-loop provider.",
            ),
            Node(
                package="mc_resource_authority",
                executable="resource_authority",
                name="mc_resource_authority",
                output="screen",
            ),
            Node(
                package="mc_world_state",
                executable="world_state",
                name="mc_world_state",
                output="screen",
            ),
            Node(
                package="seattle_lab",
                executable="mc_embodied_skills",
                name="mc_embodied_skills",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_embodied_skills")),
                parameters=[
                    {
                        "allow_fake_success": LaunchConfiguration("embodied_allow_fake_success"),
                    }
                ],
            ),
            Node(
                package="mc_ai_bt",
                executable="ai_bt_confirmation",
                name="mc_ai_bt_human_confirmation",
                output="screen",
            ),
            Node(
                package="mc_ai_bt",
                executable="ai_bt",
                output="screen",
                parameters=[
                    {
                        "planner_backend": LaunchConfiguration("planner_backend"),
                        "planner_model_name": LaunchConfiguration("planner_model_name"),
                        "planner_temperature": LaunchConfiguration("planner_temperature"),
                        "planner_timeout": LaunchConfiguration("planner_timeout"),
                        "mission_journal_path": LaunchConfiguration("mission_journal_path"),
                    }
                ],
            ),
        ]
    )
