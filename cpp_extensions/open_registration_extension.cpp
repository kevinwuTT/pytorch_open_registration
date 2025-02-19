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

#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h> // abs_stub

#include <list>

// Order of header matters here because c10 has a Layout type as well. If this is before Aten headers, there will be errors about ambiguity.
#include "ttnn/device.hpp"
#include "ttnn/operations/creation.hpp"
#include "tt-metalium/small_vector.hpp"
#include "ttnn/common/queue_id.hpp"
#include <tt-metalium/bfloat16.hpp>

#include "TtnnOpaqueTensorImpl.h"
#include "tt-metalium/event.hpp"
#include "ttnn/async_runtime.hpp"

#include <torch/csrc/utils/pybind.h>


#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// use glog instead maybe?
#define LOGGING(s) std::cout << __FILE_NAME__ << "(" << __LINE__ << ")" << "(" << __FUNCTION__ << ")" << ": " << s << std::endl

namespace {

void abs_kernel(at::TensorIteratorBase& iter) {
  // empty because we don't need it, but it has to be defined
}

} // namespace

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);
}

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
  DummyDeviceGuardImpl() {
    LOGGING("");
  }
  explicit DummyDeviceGuardImpl(c10::DeviceType t) {
    LOGGING("");
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }
  at::DeviceType type() const override {
    LOGGING("");
    return at::DeviceType::PrivateUse1;
  }
  at::Device exchangeDevice(at::Device d) const override {
    LOGGING("");
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
    LOGGING("");
    return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
  }
  void setDevice(at::Device d) const override {
    LOGGING("");
    TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
    at::Device current_device = getDevice();
    if (current_device != d) {
      CURR_DEVICE = d.index();
    }
  }
  void uncheckedSetDevice(at::Device d) const noexcept override {
    LOGGING("");
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
    LOGGING("");
    TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    LOGGING("");
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

  IDevice* get_ttnn_device() {
    LOGGING("");
    if (!ttnn_device) {
      ttnn_device = &ttnn::open_device(0);
    }
    return ttnn_device;
  }

  static IDevice * ttnn_device;
 private:
  c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
};

IDevice* DummyGuard::ttnn_device = nullptr;

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
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt) {
  LOGGING("");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // Check for value to be safe
  auto dtype = c10::scalarTypeToTypeMeta(dtype_opt.value());
  // Shape
  at::Device device = device_opt.value();
  LOGGING(device);
  DummyGuard device_guard(device);
  IDevice* ttnn_device = device_guard.get_ttnn_device();
  LOGGING(size);
  ttnn::SmallVector<uint32_t> small_vector(size.begin(), size.end());
  auto tensor = ttnn::empty(ttnn::Shape(small_vector),
    ttnn::DataType::BFLOAT16,
    ttnn::TILE_LAYOUT,
    ttnn_device,
    MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
  LOGGING("");
  auto tensor_ret = at::detail::make_tensor<at::TtnnTensorImpl>(private_use_ks, dtype, device, tensor);
  LOGGING("");
  return tensor_ret;
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  LOGGING("");
  return self;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  LOGGING("");
  // Only supports cpu ==> ttnn or ttnn ==> cpu
  // Check direction of copy
  if (self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1) {
    DummyGuard device_guard(at::device_of(dst).value());
    // TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
    // TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

    // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
    LOGGING(self.sizes());
    LOGGING(dst.sizes());
    TORCH_CHECK(self.sizes() == dst.sizes());
    TORCH_CHECK(self.scalar_type() == dst.scalar_type());
    TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

    LOGGING(self);
    LOGGING(self.dtype());

    at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(dst.unsafeGetTensorImpl());
    auto dst_tensor = tensor_impl->get_ttnn_tensor();
    auto padded_shape = dst_tensor.get_padded_shape();

    IDevice* ttnn_device = device_guard.get_ttnn_device();
    LOGGING(dst_tensor.write_to_string());
    LOGGING("");
    tt::tt_metal::memcpy(dst_tensor, self.storage().data_ptr().get());
    LOGGING(dst_tensor.write_to_string());
  }
  else if (self.device().type() == c10::DeviceType::PrivateUse1 && dst.is_cpu()) {
    DummyGuard device_guard(at::device_of(self).value());

    // same as custom_resize_?
    at::TensorImpl* dst_tensor_impl = dst.unsafeGetTensorImpl();
    LOGGING(dst_tensor_impl->storage_initialized());
    at::IntArrayRef size = self.sizes();
    dst_tensor_impl->set_sizes_contiguous(size);
    const auto itemsize = dst_tensor_impl->dtype().itemsize();
    const auto offset = dst_tensor_impl->storage_offset();
    const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
    LOGGING(storage_size);
    LOGGING(dst_tensor_impl->numel());
    at::native::maybe_resize_storage_cpu(dst_tensor_impl, storage_size);

    LOGGING(dst_tensor_impl->storage_initialized());

    // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
    LOGGING(self.sizes());
    LOGGING(dst.sizes());
    TORCH_CHECK(self.sizes() == dst.sizes());
    TORCH_CHECK(self.scalar_type() == dst.scalar_type());

    // ttnn tensor has contiguous check too
    LOGGING(self.is_contiguous());
    LOGGING(dst.is_contiguous());
    TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

    at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
    auto self_tensor = tensor_impl->get_ttnn_tensor();
    self_tensor.print();
    LOGGING(self_tensor.get_padded_shape());
    LOGGING(self_tensor.logical_shape());
    auto volume = self_tensor.volume();
    auto element_size = self_tensor.element_size();
    auto logical_volume = self_tensor.get_logical_volume();
    LOGGING(volume);

    LOGGING("");
    // better type for bytes?
    uint8_t* temp_host_ptr = new uint8_t[volume * element_size];
    tt::tt_metal::memcpy(temp_host_ptr, self_tensor);
    // use better copy?
    memcpy(dst.storage().data_ptr().get(), temp_host_ptr, logical_volume * element_size);
    delete temp_host_ptr;
  }

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
  // Check for value to be safe
  auto dtype = c10::scalarTypeToTypeMeta(dtype_opt.value());

  at::Device device = device_opt.value();
  LOGGING(device);
  DummyGuard device_guard(device);
  IDevice* ttnn_device = device_guard.get_ttnn_device();
  ttnn::SmallVector<uint32_t> small_vector(size.begin(), size.end());
  auto tensor = ttnn::empty(ttnn::Shape(small_vector),
    ttnn::DataType::BFLOAT16,
    ttnn::TILE_LAYOUT,
    ttnn_device,
    MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
  LOGGING(tensor.write_to_string());
  auto tensor_ret = at::detail::make_tensor<at::TtnnTensorImpl>(private_use_ks, dtype, device, tensor);
  return tensor_ret;
}

at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  LOGGING("");
  LOGGING(self.device().type());
  LOGGING(out.device().type());
  at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(self.unsafeGetTensorImpl());
  LOGGING("");
  auto ttnn_tensor = tensor_impl->get_ttnn_tensor();
  LOGGING("");
  ttnn_tensor.print();

  LOGGING("");
  auto result = ttnn::abs(ttnn_tensor);

  result.print();

  // this might belong somewhere else?
  LOGGING("");
  at::TtnnTensorImpl* out_tensor_impl = static_cast<at::TtnnTensorImpl*>(out.unsafeGetTensorImpl());
  LOGGING(out.device().type());
  out_tensor_impl->set_sizes_and_strides_as(self);
  LOGGING(out.device().type());
  LOGGING("");

  auto out_ttnn_tensor = out_tensor_impl->get_ttnn_tensor();
  LOGGING("");
  out_tensor_impl->set_ttnn_tensor(result);
  LOGGING("");
  
  // return at::native::abs_out(self, out);
  return out;
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
  m.impl("aten::empty_strided", &custom_empty_strided);
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("_copy_from", &custom__copy_from);
  m.impl("abs.out", &custom_abs_out); 
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device(int idx) {
  LOGGING("");
  auto device = c10::Device(c10::DeviceType::PrivateUse1, idx);
  return device;
}

// How to automatically close device without explicit calling?
void close_custom_device(c10::Device device) {
  LOGGING("");
  DummyGuard device_guard(device);
  IDevice* ttnn_device = device_guard.get_ttnn_device();
  TORCH_INTERNAL_ASSERT(ttnn_device != nullptr);
  ttnn::close_device(*ttnn_device);
  ttnn_device = nullptr;
}

// Get underlying ttnn tensor
ttnn::Tensor get_ttnn_tensor(at::Tensor& tensor) {
  // Check if this cast fails
  at::TtnnTensorImpl* tensor_impl = static_cast<at::TtnnTensorImpl*>(tensor.unsafeGetTensorImpl());
  auto ttnn_tensor = tensor_impl->get_ttnn_tensor();
  return ttnn_tensor;
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("close_custom_device", &close_custom_device, "close custom device object");
    m.def("get_ttnn_tensor", &get_ttnn_tensor, "get underlying ttnn tensor");
}
