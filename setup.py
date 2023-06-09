"""Setup file."""
from setuptools import setup
import versioneer

if __name__ == "__main__":
    setup(
        name="caf.distribute",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
