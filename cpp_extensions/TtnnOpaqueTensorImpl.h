#pragma once

#include <ATen/OpaqueTensorImpl.h>
#include "ttnn/tensor/tensor.hpp"
#include <iostream>
#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// use glog instead maybe?
#define LOGGING(s) std::cout << __FILE_NAME__ << "(" << __LINE__ << ")" << "(" << __FUNCTION__ << ")" << ": " << s << std::endl

namespace at {

struct TtnnTensorImpl : public TensorImpl {
  TtnnTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      ttnn::Tensor& ttnn_tensor) : TensorImpl(key_set, data_type, device), ttnn_tensor_(ttnn_tensor) {
        LOGGING(device.type());
        auto view = ttnn_tensor_.get_logical_shape().view();
        std::vector<int64_t> view_int64;
        std::copy(view.begin(), view.end(), std::back_inserter(view_int64));
        IntArrayRef int_array_ref(&(*view_int64.begin()), &(*view_int64.end()));
        sizes_and_strides_.set_sizes(int_array_ref);
      }

  void set_sizes_and_strides(const IntArrayRef& int_array_ref) {
      sizes_and_strides_.set_sizes(int_array_ref);
  }

  void set_sizes_and_strides_as(const at::Tensor& the_template) {
    sizes_and_strides_.set_sizes(the_template.sizes());
  }

  ttnn::Tensor get_ttnn_tensor() {
    return ttnn_tensor_;
  }

  void set_ttnn_tensor(const ttnn::Tensor& tensor) {
    ttnn_tensor_ = tensor;
  }

  private:
    ttnn::Tensor ttnn_tensor_;
};

} // namespace at