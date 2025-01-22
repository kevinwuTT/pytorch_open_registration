import torch
import torch.utils.cpp_extension
from torch.overrides import TorchFunctionMode
import os
from pathlib import Path

os.environ["CXX"] = "clang++-17"
# os.environ["CFLAGS"] = "-std=c++20 -stdlib=libc++"

# c++20 needed for __cpp_concepts
# c++20 -std=libc++ needed for source_location
site_packages = Path("/home/ubuntu/virtualenv/torch_ttnn/lib/python3.8/site-packages/")
tt_metal_home = Path("/home/ubuntu/repo/tt-metal")
ttnn_include_paths = [
    tt_metal_home,
    tt_metal_home / Path("tt_metal"),
    tt_metal_home / Path("tt_metal/third_party/umd"),
    tt_metal_home / Path("tt_metal/third_party/fmt"),
    tt_metal_home / Path("tt_metal/hw/inc/wormhole"),
    tt_metal_home / Path("tt_metal/hw/inc/wormhole/wormhole_b0_defines"),
    tt_metal_home / Path("tt_metal/hw/inc/"),
    tt_metal_home / Path("tt_metal/third_party/umd/src/firmware/riscv/wormhole"),
    tt_metal_home / Path("tt_metal/third_party/umd/device"),
    tt_metal_home / Path(".cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include"),
    tt_metal_home / Path(".cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include/magic_enum"),
    tt_metal_home / Path(".cpmcache/boost_core/e679bef5c160cf29d0f37d549881dc5f5a58c332/include"),
    tt_metal_home / Path(".cpmcache/boost_container/5fb02b14b46d0d84e7a0ce09e2ea5e963d5d93bd/include"),
    tt_metal_home / Path(".cpmcache/boost_config/0bad5ba3b48288a243894aa801ed6eccbef70b60/include"),
    tt_metal_home / Path(".cpmcache/boost_move/c59effd88face3140123440bc5425ee60328f08d/include"),
    tt_metal_home / Path(".cpmcache/boost_intrusive/4a7bf962355d8580809cea3c68f55bbaaa746e64/include"),
    tt_metal_home / Path(".cpmcache/boost_assert/3ab1f6f9db9a884ad9a641164dbb6589a5aa7e2d/include"),
    tt_metal_home / Path("ttnn/cpp"),
    tt_metal_home / Path("ttnn/cpp/ttnn/deprecated"),
    tt_metal_home / Path("tt_metal/third_party/magic_enum"),
    tt_metal_home / Path("tt_metal/third_party/umd/device/api"),
    tt_metal_home / Path("tt_metal/hostdevcommon/api"),
    tt_metal_home / Path(".cpmcache/json/230202b6f5267cbf0c8e5a2f17301964d95f83ff/include"),
    tt_metal_home / Path(".cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f"),
    tt_metal_home / Path(".cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include"),
    tt_metal_home / Path("tt_metal/third_party/tracy/public"),
    tt_metal_home / Path("tt_metal/include"),
    ]
ttnn_include_paths = [str(p) for p in ttnn_include_paths]

# Load the C++ extension containing your custom kernels.
tt_metal_lib_paths = [
    tt_metal_home / Path("build/lib"),
    # tt_metal_home / Path("build/_deps/fmt-build")
]
# Should we include rpath? Tradeoff between having to call LD_LIBRARY_PATH
tt_metal_lib_paths = ["-L" + str(p) + " -Wl,-rpath=" + str(p) for p in tt_metal_lib_paths]
tt_metal_libs = [
    "tt_metal",
    "c++",
    ":_ttnn.so",
    "device",
]
tt_metal_libs = ["-l" + p for p in tt_metal_libs]

foo_module = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[
        "cpp_extensions/open_registration_extension.cpp",
    ],
    extra_include_paths=["cpp_extensions"] + ttnn_include_paths,
    extra_cflags=[ "-g", "-DFMT_HEADER_ONLY"],
    extra_ldflags=tt_metal_lib_paths + tt_metal_libs,
    verbose=True,
)

print('Loaded custom extension.')

# The user will globally enable the below mode when calling this API
def enable_foo_device():
    m = TtnnDeviceMode()
    m.__enter__()
    # If you want the mode to never be disabled, then this function shouldn't return anything.
    return m

# This is a simple TorchFunctionMode class that:
# (a) Intercepts all torch.* calls
# (b) Checks for kwargs of the form `device="foo:i"`
# (c) Turns those into custom device objects: `device=foo_module.custom_device(i)`
# (d) Forwards the call along into pytorch.
class TtnnDeviceMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if 'device' in kwargs and 'ttnn' in kwargs['device']:
            device_and_idx = kwargs['device'].split(':')
            if len(device_and_idx) == 1:
                # Case 1: No index specified
                kwargs['device'] = foo_module.custom_device()
            else:
                # Case 2: The user specified a device index.
                device_idx = int(device_and_idx[1])
                kwargs['device'] = foo_module.custom_device(device_idx)
        with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)
