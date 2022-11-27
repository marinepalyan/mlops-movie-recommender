import setuptools

about = dict()

with open("./src/movie_predictor/__version__.py", "r") as fp:
    exec(fp.read(), about)

with open("requirements.txt", "r") as requirements_file:
    reqs = requirements_file.read().splitlines()


packages = list()

for req in reqs:
    if req.startswith("git+ssh"):
        packages.append(req.split("/")[-1].split("@")[0] + " @ " + req)
    else:
        packages.append(req)

setuptools.setup(
    name="{{ cookiecutter.model_name }}",
    version=about["__version__"],
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="src"),
    long_description=open("README.md").read(),
    install_requires=packages,
)