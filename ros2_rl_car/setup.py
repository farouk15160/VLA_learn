from setuptools import find_packages, setup


PACKAGE_NAME = "ros2_rl_car"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", ["launch/simulation.launch.py"]),
        (f"share/{PACKAGE_NAME}/config", ["config/default.json"]),
        (
            f"share/{PACKAGE_NAME}/assets/worlds",
            ["assets/worlds/centerline.csv", "assets/worlds/alternating_track.world"],
        ),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=False,
    maintainer="VLA Learn",
    maintainer_email="maintainer@example.invalid",
    description="CPU-first PPO driving in ROS 2 Humble and Gazebo Classic 11",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rl-car = ros2_rl_car.cli:main",
            "rl-car-train = ros2_rl_car.learning.trainer:main",
            "rl-car-evaluate = ros2_rl_car.evaluation.runner:main",
            "rl-car-generate = ros2_rl_car.sim.world:main",
        ]
    },
)
