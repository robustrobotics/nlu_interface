from setuptools import find_packages, setup

setup(
    name="nlu_interface",
    version="0.0.1",
    url="",
    author="",
    author_email="",
    description="Language interface.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.pddl"]},
    install_requires=[
        "spark_dsg",
    ],
)
