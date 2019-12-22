import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="texture-gen-aremath",
    version="0.0.2",
    author="Ross Mawhorter",
    author_email="rmawhorter@g.hmc.edu",
    description="Image processing and texture generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aremath/texture_gen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
    python_requires=">=3.6",
)
