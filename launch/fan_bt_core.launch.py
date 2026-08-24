from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
