from glob import glob

from setuptools import find_packages, setup

package_name = "rozart_residual"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lopatoj",
    maintainer_email="justin@lopa.to",
    description="Residual policy deployment and on-robot fine-tuning for Rozart through ros2_control",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "residual_node = rozart_residual.residual_node:main",
        ],
    },
)
