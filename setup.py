import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simd-structts",
    version="0.1.0",
    author="Vladimir Shulyak",
    author_email="vladimir@shulyak.net",
    description="SIMD StuctTS Model with various backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vshulyak/simd-structts",
    packages=setuptools.find_packages(),
    install_requires=[
        'statsmodels>=0.11.1',
        'simdkalman>=1.0.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
