import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spellcheck-pkg", # Replace with your own username
    version="0.0.1",
    author="Leonardo Araujo",
    author_email="leolca@gmail.com",
    description="Spell Check uses frequency of occurrence and keyboard or phonetic distance to find suggestions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leolca/spellcheck",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
