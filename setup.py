"""Build native-resource NHSMM wheels with an OS/architecture-specific tag."""

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class NativePlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # Contains native Rust ELF resources; never advertise py3-none-any.
        self.root_is_pure = False

    def get_tag(self):
        # Rust ABI depends on platform, not the Python interpreter ABI.
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(cmdclass={"bdist_wheel": NativePlatformWheel})
