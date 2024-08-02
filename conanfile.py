from conan import ConanFile


class Recipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps", "VirtualRunEnv"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_gpu": [True, False],
        "with_npu": [True, False],
    }

    default_options = {
        "shared": False,
        "fPIC": True,
        "with_gpu": True,
        "with_npu": True,
    }

    def layout(self):
        self.folders.generators = "conan"

    def requirements(self):
        self.requires("fmt/10.2.1")
        self.requires("edgerunner/0.1.2")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

        self.options["edgerunner"].with_gpu = self.options.with_gpu
        self.options["edgerunner"].with_npu = self.options.with_npu
        self.options["edgerunner"].with_tflite = False

    def build_requirements(self):
        self.test_requires("catch2/3.6.0")
