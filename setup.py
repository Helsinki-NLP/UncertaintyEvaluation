import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "Click",
    "setuptools",
    "transformers"
]

tests_require = [
    'pytest'
]

setuptools.setup(
    name="unceval",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="Hande Celikkanat, Sami Virpioja",
    author_email="sami.virpioja@helsinki.fi",
    description="Evaluation tool for uncertainty-aware language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helsinki-NLP/UncertaintyEvaluation",
    package_dir={"": "src"},
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'unceval = unceval.main:cli',
        ],
    },
)
