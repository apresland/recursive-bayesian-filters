import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recursive_bayesian_filters", # Replace with your own username
    version="0.0.1",
    author="Andrew Presland",
    author_email="andrew.presland@gmail.com",
    description="A set of recursive localization filters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apresland/recursive_baysian_filters",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)