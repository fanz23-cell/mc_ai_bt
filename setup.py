import os
from glob import glob

from setuptools import find_packages, setup

package_name = "mc_ai_bt"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yue Feng",
    maintainer_email="yue.feng@mindchildren.com",
    description="AI-BT mission manager, planner validator and execution shell.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "ai_bt = mc_ai_bt.node:main",
            "ai_bt_doctor = mc_ai_bt.ros_doctor:main",
            "ai_bt_local_smoke = mc_ai_bt.local_smoke:main",
            "ai_bt_local_fake_ros_smoke = mc_ai_bt.ros_smoke:main",
            "ai_bt_fake_skill_servers = mc_ai_bt.fake_skill_servers:main",
        ],
    },
)
