from setuptools import setup, find_packages

setup(
    name = "sliding-mode-demo",
    description = "sliding mode controller demo",
    version = "0.1",
    author = 'Maksim Surov',
    author_email = "surov.m.o@gmail.com",
    python_requires = ">=3.9",
    install_requires = [
        "numpy",
        "matplotlib",
        "tk",
        "scipy",
        "wheel",
    ],
    packages = find_packages(where="src"),
    package_dir={"": "sliding_mode_demo"},
    entry_points = {
        "console_scripts": [
        ]
    },
    include_package_data = True,
)
