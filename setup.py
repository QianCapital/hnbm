from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


def _parse_requirements(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pip") or line.startswith("setuptools"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="hnbm",
    version="0.1.0",
    author="Qian Capital",
    author_email="samson.qian@qiancapital.com",
    packages=["hnbm"],
    url="https://github.com/qiancapital-dev/hnbm",
    license="MIT",
    description="Heterogeneous Newton Boosting Machine",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=_parse_requirements("requirements.txt"),
    python_requires=">=3.8",
    keywords="boosting gradient-boosting hnbm heterogeneous-newton",
)
