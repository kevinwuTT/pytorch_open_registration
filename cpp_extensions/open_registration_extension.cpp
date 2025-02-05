#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/abs_native.h>

#include <iostream>
#include <string.h>

// #include "ttnn/device.hpp"

#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h> // abs_stub

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// use glog instead maybe?
#define LOGGING(s) std::cout << __FILE_NAME__ << "(" << __LINE__ << ")" << "(" << __FUNCTION__ << ")" << ": " << s << std::endl

static uint64_t abs_counter = 0;

namespace {

// Using the simplest way to obtain continuous Tensor data and process it.
// This is a demo for using operand API, and you can add more complex logic
// for input and output tensor based on your custom device kernel.

// Taken from old version https://github.com/pytorch/pytorch/blob/f47aac6/test/cpp_extensions/open_registration_extension.cpp
void abs_kernel(at::TensorIteratorBase& iter) {
  LOGGING("");

  // Abs only have a input tensor and a output tensor.
  auto& output_operand = iter.operand(0);
  auto& input_operand = iter.operand(1);
  auto& output_tensor_base = output_operand.tensor_base();
  auto& input_tensor_base = input_operand.tensor_base();
  TORCH_CHECK(!input_operand.original_tensor_base().defined(),
    "input original tensor is defined.");
  TORCH_CHECK(!output_operand.original_tensor_base().defined(),
    "output original tensor is defined.");
  // For easy test, only accept contiguous input tensor for calculate.
  auto memory_format = input_tensor_base.suggest_memory_format();
  TORCH_CHECK(input_tensor_base.is_contiguous(memory_format),
    "Input tensor need be contiguous.");
  // Add necessary restrictions to ensure the security of the demo.
  TORCH_CHECK(input_tensor_base.sizes() == output_tensor_base.sizes(),
    "Intput and output tensor size are not equal.");
  // Common dtype is calculate in TensorIteratorBase.
  TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float,
    "Only support float type.")
  // Using for loop for abs calculate.
  auto abs_function = [](float* output_ptr, const float* input_ptr,
                         const int64_t NUM) {
    for (int64_t i = 0; i < NUM; ++i) {
      *(output_ptr + i) = std::abs(*(input_ptr + i));
    }
  };
  // To simplify the logic of the test demo code,
  // we only use contiguous tensor to calculate on device side.
  // And using input tensor memory format.
  if (iter.is_contiguous()) {
    // Add for will_resize flag check. You can convert to differernt
    // tensor memory format when will_resize is True.
    // If TensorIteratorConfig resize_outputs_ flag is true, and there are two
    // situations:
    // 1) Out tensor is undefined, and TensorIterator set will_resize to true;
    // 2) Out tensor is defined and tensor size is not equal to input tensor size;
    //    TensorIterator set will_resize to true, and call set_output_raw_strided
    //    to resize output tensor.
    // When output operand will_resize flag is ture, dummy
    // device can convert tensor to dummy device preferred memory format.
    // Here we don't convert tensor memory format, because it will become complex
    // when dummy device want keep same memory format for training network.
    TORCH_CHECK(output_operand.will_resize,
      "output operand will_resize flag need be True.");
    abs_function((float*)iter.data_ptr(0), (float*)iter.data_ptr(1), iter.numel());
  } else {
    // Stride copy is not support for foo device, using cpu device instead.
    // For abs op, the last situation is: output tensor is not contiguous with
    // operand will_resize is False.
    TORCH_CHECK(!output_operand.will_resize, "output operand will_resize is True.");
    // Get a contiguous tensor with input memory format.
    at::Tensor output = at::empty(output_tensor_base.sizes(),
                                  input_tensor_base.options()
                                                   .memory_format(memory_format));
    // For structured op which inheried from TensorIteratorBase, maybe you need to
    // call set_output_raw_strided function to update output stored in op sturctured.
    // abs op is no need to do this.
    output_operand.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(std::in_place, output));
    abs_function((float*)output_operand.tensor_base().mutable_data_ptr(),
                 (float*)iter.data_ptr(1), iter.numel());
    // Copy tensor base to original tensor base, and keep same scalar type and
    // stride with cpu and gpu.
    if (output_operand.original_tensor_base().defined() &&
        !output_operand.original_tensor_base().is_same(output_operand.tensor_base())) {
      output_operand.original_tensor().copy_(output_operand.tensor());
      output_operand.restore_original_tensor();
    }
  }
}

} // namespace

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);
}

// This file contains the heavy lifting to add a new C++ backend
// and integrate it directly into the PyTorch backend. It mainly involves:
//
// (1) Writing a custom allocator and registering it to pytorch
//     (see DummyCustomAllocator)
// (2) Writing a custom device guard, registering it to pytorch,
//     and using the device guard in kernels
//     (see DummyDeviceGuard)
// (3) Writing a custom aten::empty.memory_format function


// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << __FILENAME__ << " Custom aten::add.Tensor() called!" << std::endl;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// =====================================
// ========= Custom Allocators =========
// =====================================

// PyTorch provides an API for registering custom allocators for your device.
// You can create one by inheriting from the at::Allocator class,
// and registering your allocator for the particular device type
// (PrivateUse1 for open registration devices)

// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  // at::DataPtr allocate(size_t nbytes) override {
  at::DataPtr allocate(size_t nbytes) const override {
    LOGGING("");
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  // void copy_data(void* dest, const void* src, std::size_t count) const override {
  //   std::cout << __FILENAME__ << " Custom copy copy_data() called!" << std::endl;
  //   std::memcpy(dest, src, count);
  // }

  static void ReportAndDelete(void* ptr) {
    LOGGING("");
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
  // at::DeleterFnPtr raw_deleter() {
    return &ReportAndDelete;
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

// =====================================
// ============= Device Guards =========
// =====================================

// PyTorch has an API for registering device guards.
// Device guards can be used to set the current "active" device,
// and e.g. error if the user provides an invalid device index.
//
// If your device doesn't support indices (e.g. foo:0 vs. foo:1),
// then the guards probably aren't needed.
//
// You can use it by creating a DeviceGuard class, registering it
// in PyTorch, and invoking the device guard before any kernels are called.
// For a more full-featured example of a device guard,
// check out the code at c10/cuda/CUDAGuard.h

// Represents the current "active" device.
// The dummy device guard registered below is meant to show how a backend
// can integrate custom device guard with pytorch.
// For something like cuda this represents the current active cuda device,
// which is directly set using the cuda API calls cudaGetDevice/cudaSetDevice.
static uint16_t CURR_DEVICE = -1;

// Create and register a dummy device guard.
struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
  DummyDeviceGuardImpl() {}
  explicit DummyDeviceGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }
  at::DeviceType type() const override {
    return at::DeviceType::PrivateUse1;
  }
  at::Device exchangeDevice(at::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    at::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      // "set the active device"
      CURR_DEVICE = d.index();
    }
    return old_device;
  }
  at::Device getDevice() const override {
    return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
  }
  void setDevice(at::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    at::Device current_device = getDevice();
    if (current_device != d) {
      CURR_DEVICE = d.index();
    }
  }
  void uncheckedSetDevice(at::Device d) const noexcept override {
    auto current_device = getDevice();
    if (current_device != d) {
      CURR_DEVICE = d.index();
    }
  }
  at::Stream getStream(at::Device d) const noexcept override {
    // no-op
    return at::Stream(at::Stream::DEFAULT, d);
  }
  // NB: These do NOT set the current device
  at::Stream exchangeStream(at::Stream) const noexcept override {
    // no-op
    return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE));
  }
  at::DeviceIndex deviceCount() const noexcept override {
    // Hardcoding the number of "valid" devices here at 2.
    return 2;
  }

  // Event-related functions
  void record(
      void** /*event*/,
      const at::Stream& /*stream*/,
      const at::DeviceIndex /*device_index*/,
      const c10::EventFlag /*flag*/) const override {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
  }
  void block(void* /*event*/, const at::Stream& /*stream*/) const override {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
  }
  void destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/)
      const noexcept override {}

  // Stream-related functions
  bool queryStream(const at::Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const at::Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};

struct DummyGuard {
  explicit DummyGuard() = delete;
  explicit DummyGuard(at::DeviceIndex device_index) : guard_(device_index) {}
  explicit DummyGuard(at::Device device) : guard_(device) {}
  DummyGuard(const DummyGuard&) = delete;
  DummyGuard& operator=(const DummyGuard&) = delete;
  DummyGuard(DummyGuard&& other) = delete;
  DummyGuard& operator=(DummyGuard&& other) = delete;

  void set_device(at::Device device) {
    guard_.set_device(device);
  }

  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  void set_index(at::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  at::Device original_device() const {
    return guard_.original_device();
  }

  at::Device current_device() const {
    return guard_.current_device();
  }

 private:
  c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);


// =====================================
// ============= KERNELS ===============
// =====================================

// basic dummy empty function, so we can directly construct tensors on the custom device
// This dummy test device will just use the CPU allocator, and ignores pinned memory.
//
// Note: this kernel is very simple because our "custom device" just uses the normal TensorImpl object
// to store data under the hood.
// In PyTorch core today, both cpu and cuda are implemented with an ordinary TensorImpl class.
// Sometimes, backends prefer to subclass TensorImpl in order to store extra information.
// If this is the case, then this kernel is where you'll be responsible for creating and returning
// a fresh at::Tensor object, that properly stores a TensorImpl of your subclass.
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  // const at::OptionalDeviceGuard device_guard(device);
  LOGGING("");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto empty_generic = at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
  return empty_generic;
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  // const at::OptionalDeviceGuard device_guard(at::device_of(self));
  LOGGING("");
  // return at::native::fill_(self, value);
  return self;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  LOGGING("");
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}

at::Tensor custom_empty_strided(c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                c10::optional<at::ScalarType> dtype_opt,
                                c10::optional<at::Layout> layout_opt,
                                c10::optional<at::Device> device_opt,
                                c10::optional<bool> pin_memory_opt) {

  LOGGING("");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  LOGGING("");
  return at::native::abs_out(self, out);
}

const at::Tensor& custom_resize_(const at::Tensor& self, at::IntArrayRef size,
                          std::optional<at::MemoryFormat> optional_memory_format) {
  LOGGING("");
  at::TensorImpl* tensor_impl = self.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(size);
  const auto itemsize = tensor_impl->dtype().itemsize();
  const auto offset = tensor_impl->storage_offset();
  const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
  // Dummy device is using cpu allocator, so here just call cpu
  // function maybe_resize_storage_cpu in aten/src/ATen/native/Resize.h
  // to get a sufficient memory space.
  at::native::maybe_resize_storage_cpu(tensor_impl, storage_size);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != at::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    tensor_impl->empty_tensor_restride(memory_format);
  }
  return self;
}

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("fill_.Scalar", &custom_fill__scalar);
  m.impl("_copy_from", &custom__copy_from);
  m.impl("aten::empty_strided", &custom_empty_strided);
  m.impl("abs.out", &custom_abs_out); 
  m.impl("resize_", &custom_resize_);
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device(int idx) {
  return c10::Device(c10::DeviceType::PrivateUse1, idx);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
}
