from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    core_launch = PathJoinSubstitution(
        [FindPackageShare("mc_ai_bt"), "launch", "fan_bt_core.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner_backend",
                default_value="bootstrap",
                description="Planner backend passed to fan_bt_core.launch.py.",
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
                "fake_delay_sec",
                default_value="0.05",
                description="Fake action completion delay in seconds.",
            ),
            DeclareLaunchArgument(
                "fake_object_name",
                default_value="cup",
                description="Fake object name published to mc_world_state.",
            ),
            DeclareLaunchArgument(
                "fake_object_score",
                default_value="0.9",
                description="Fake object confidence score.",
            ),
            DeclareLaunchArgument(
                "fake_visible_people_count",
                default_value="1",
                description="Number of fake visible people published to mc_world_state.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(core_launch),
                launch_arguments={
                    "planner_backend": LaunchConfiguration("planner_backend"),
                    "planner_model_name": LaunchConfiguration("planner_model_name"),
                    "planner_temperature": LaunchConfiguration("planner_temperature"),
                    "planner_timeout": LaunchConfiguration("planner_timeout"),
                    "mission_journal_path": LaunchConfiguration("mission_journal_path"),
                }.items(),
            ),
            Node(
                package="mc_ai_bt",
                executable="ai_bt_fake_skill_servers",
                name="mc_ai_bt_fake_skill_servers",
                output="screen",
                parameters=[
                    {
                        "delay_sec": LaunchConfiguration("fake_delay_sec"),
                        "fake_object_name": LaunchConfiguration("fake_object_name"),
                        "fake_object_score": LaunchConfiguration("fake_object_score"),
                        "fake_visible_people_count": LaunchConfiguration("fake_visible_people_count"),
                    }
                ],
            ),
        ]
    )
