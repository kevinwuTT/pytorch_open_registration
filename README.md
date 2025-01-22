# pytorch_open_registration_example
Example of using pytorch's open device registration API. It covers:

(1) Writing custom kernels in C++, and registering them to the PyTorch dispatcher

(2) Providing a user API for your custom device, so users can invoke the custom code using `torch.foo(..., device="custom_device")`

(3) Registering a custom memory allocator

(4) Registering a custom device guard

This repo should be run using the latest pytorch nightly. If it fails on the latest nightly, please post an issue and I'll take a look!

# Build pytorch with clang and libc++
Pytorch installed from the official wheels were built with GCC and libstdc++. tt-metal uses clang (version 17 as of this writing) and c++20. To support new C++20 libraries, we link with libc++ that is included with clang instead of stdc++ that’s installed in the system. This can complicate linking tt-metal libraries with Pytorch because of the way some symbols are named. This typically results in “undefined symbol” errors during runtime, not necessarily compile/link time. 

Examples:

* https://github.com/pytorch/glow/issues/5713
* When trying to link this test https://github.com/bdhirsh/pytorch_open_registration_example with tt-metal, I’ve encountered an error like below when using clang-17 with c++20.
custom_device_extension.so: undefined symbol: `_ZN3c106detail14torchCheckFailEPKcS2_jRKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE`

To fix this we will need to build pytorch with clang and libc++

Obtain the following in some way:

* Ubuntu 20.04
    * Python 3.8
* clang-17
* cmake >= 3.18.0

1. Create a fresh venv for this purpose. 
    ```
    python3 -m venv ~/venv/torch_clang
    . ~/venv/torch_clang/bin/activate
    ```

1. Clone pytorch and checkout the version matching current tt-metal builds
    ```
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git checkout tags/v2.2.1
    ```

1. Because we’re using an older version of Python, we need to manually install some dependencies before installing from requirements.txt
    ```
    pip install networkx==2.8.8
    pip install -r requirements.txt
    ```

1. Install some more system dependencies. Note that libomp-17 matches the same version as clang-17.
    ```
    sudo apt install libomp-17-dev
    ```

1. Build and install pytorch with clang-17, c++20, and libc++
    ```
    CC=clang-17 CXX=clang++-17 CXXFLAGS="-std=c++20 -stdlib=libc++" python setup.py install
    ```

That should be enough. Pytorch should now be installed in the current virtual environment.
