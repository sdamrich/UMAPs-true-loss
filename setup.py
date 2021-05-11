from setuptools import setup



configuration = {
    "name": "umapns",
    "version": "0.1",
    "description": "Loss logging extentions to UMAP",
    "long_description_content_type": "text/x-rst",
    "keywords": "dimension reduction t-sne manifold",
    "license": "MIT",
    "packages": ["umapns"],
    "install_requires": [
        "numpy >= 1.17",
        "scikit-learn >= 0.22",
        "scipy >= 1.0",
        "numba >= 0.49",
        "pynndescent >= 0.5",
    ],
    "extras_require": {
        "plot": [
            "pandas",
            "matplotlib",
            "datashader",
            "bokeh",
            "holoviews",
            "colorcet",
            "seaborn",
            "scikit-image",
        ],
        "parametric_umap":
        [
            "tensorflow >= 2.1"
        ]
    },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
