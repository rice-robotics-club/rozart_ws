from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("rozart_residual"), "config", "residual.yaml"]
            ),
        ),
        DeclareLaunchArgument("mode", default_value="deploy"),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("iterations", default_value="100"),
        DeclareLaunchArgument("device", default_value="cpu"),
    ]
    node = Node(
        package="rozart_residual",
        executable="residual_node",
        output="screen",
        parameters=[
            {
                name: LaunchConfiguration(name)
                for name in ("config", "mode", "checkpoint", "device")
            },
            {"iterations": ParameterValue(LaunchConfiguration("iterations"), value_type=int)},
        ],
    )
    return LaunchDescription([*args, node])
