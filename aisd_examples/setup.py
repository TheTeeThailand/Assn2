# /home/aisd/Assn2/aisd_examples/setup.py
from setuptools import setup

setup(
    name="aisd_examples",
    version="0.0.1",
    packages=["aisd_examples", "aisd_examples.envs"],
    install_requires=["gymnasium"],
    entry_points={
        "gymnasium.envs": [
            "aisd_examples/CreateRedBall-v0 = aisd_examples.envs.create_red_ball:CreateRedBallEnv"
        ]
    }
)

