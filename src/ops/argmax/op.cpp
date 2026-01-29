#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 实现argmax算子
    // 检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 检查 max_idx 和 max_val 的形状
    CHECK_ARGUMENT(max_idx->numel() == 1, "argmax: max_idx must be a single element tensor");
    CHECK_ARGUMENT(max_val->numel() == 1, "argmax: max_val must be a single element tensor");

    // 检查 max_val 和 vals 的数据类型一致
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // 检查 max_idx 的数据类型为 int64
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be of type int64");

    // 检查所有张量都是连续的
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "argmax: all tensors must be contiguous.");

    // 设置设备上下文
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
