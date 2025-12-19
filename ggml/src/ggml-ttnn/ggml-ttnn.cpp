/*
 * Copyright (c) 2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-ttnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <ttnn/device.hpp>
#include <ttnn/ttnn.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define GGML_TTNN_NAME "TTNN"

struct ggml_backend_ttnn_context {
    ttnn::Device * device = nullptr;
    int32_t device_id = -1;
};

struct ggml_backend_ttnn_device_context {
    int32_t device_id = -1;
    std::string name;
    std::string description;
    ttnn::Device * device = nullptr;
};

struct ggml_ttnn_tensor {
    std::unique_ptr<ttnn::Tensor> tensor;
};

struct ggml_backend_ttnn_buffer_context {
    void * host_ptr = nullptr;
    size_t host_size = 0;
    std::vector<ggml_ttnn_tensor *> extras;
    std::vector<ggml_tensor *> tensors;
};

static ggml_guid_t ggml_backend_ttnn_guid() {
    static ggml_guid guid = { 0x7d, 0x1a, 0x6b, 0x52, 0x2f, 0x41, 0x4c, 0x2c,
                              0x9d, 0x36, 0xf3, 0x11, 0x78, 0x11, 0xaa, 0x3b };
    return &guid;
}

static ttnn::DataType ggml_ttnn_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return ttnn::DataType::FLOAT16;
        case GGML_TYPE_BF16:
            return ttnn::DataType::BFLOAT16;
        case GGML_TYPE_F32:
            return ttnn::DataType::FLOAT32;
        case GGML_TYPE_I32:
            return ttnn::DataType::INT32;
        default:
            GGML_ABORT("TTNN backend does not support tensor type: %s", ggml_type_name(type));
    }
}

static std::vector<int64_t> ggml_ttnn_shape(const ggml_tensor * tensor) {
    std::vector<int64_t> dims;
    dims.reserve(GGML_MAX_DIMS);
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (tensor->ne[i] == 1 && dims.empty()) {
            continue;
        }
        dims.push_back(tensor->ne[i]);
    }
    if (dims.empty()) {
        dims.push_back(1);
    }
    return dims;
}

static ggml_ttnn_tensor * ggml_ttnn_get_extra(const ggml_tensor * tensor) {
    return static_cast<ggml_ttnn_tensor *>(tensor->extra);
}

static void ggml_ttnn_enable_program_cache() {
    // Best-effort: avoid overriding user config if already set.
    if (std::getenv("TTNN_CONFIG_OVERRIDES") == nullptr) {
#ifdef _WIN32
        _putenv_s("TTNN_CONFIG_OVERRIDES", "{\"enable_fast_runtime_mode\": true}");
#else
        setenv("TTNN_CONFIG_OVERRIDES", "{\"enable_fast_runtime_mode\": true}", 0);
#endif
    }
}

static ttnn::Tensor ggml_ttnn_upload_tensor(
    const ggml_tensor * tensor,
    const void * data,
    size_t size,
    ttnn::Device * device) {
    GGML_ASSERT(size == ggml_nbytes(tensor));

    const auto shape = ggml_ttnn_shape(tensor);
    const auto dtype = ggml_ttnn_dtype(tensor->type);

    auto host_tensor = ttnn::Tensor::from_borrowed_storage(
        const_cast<void *>(data),
        shape,
        ttnn::Layout::ROW_MAJOR,
        dtype);

    if (dtype == ttnn::DataType::INT32) {
        return ttnn::to_device(
            host_tensor,
            device,
            ttnn::MemoryConfig{
                ttnn::TensorMemoryLayout::INTERLEAVED,
                ttnn::BufferType::DRAM});
    }

    auto tiled_tensor = ttnn::to_layout(host_tensor, ttnn::Layout::TILE);

    return ttnn::to_device(
        tiled_tensor,
        device,
        ttnn::MemoryConfig{
            ttnn::TensorMemoryLayout::INTERLEAVED,
            ttnn::BufferType::DRAM});
}

static void ggml_ttnn_download_tensor(
    const ggml_tensor * tensor,
    const ttnn::Tensor & device_tensor,
    void * dst) {
    auto host_tensor = ttnn::from_device(device_tensor);
    auto row_major = ttnn::to_layout(host_tensor, ttnn::Layout::ROW_MAJOR);

    const size_t size = ggml_nbytes(tensor);
    std::memcpy(dst, row_major.data(), size);
}

//
// buffer
//

static void ggml_backend_ttnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;

    for (ggml_ttnn_tensor * extra : ctx->extras) {
        delete extra;
    }

    ggml_aligned_free(ctx->host_ptr);
    delete ctx;
}

static void * ggml_backend_ttnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    return ctx->host_ptr;
}

static enum ggml_status ggml_backend_ttnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        tensor->extra = tensor->view_src->extra;
        return GGML_STATUS_SUCCESS;
    }

    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto * extra = new ggml_ttnn_tensor();
    tensor->extra = extra;
    ctx->extras.push_back(extra);
    ctx->tensors.push_back(tensor);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_ttnn_buffer_memset_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size) {
    // Slow fallback: download, memset, upload.
    ggml_ttnn_tensor * extra = ggml_ttnn_get_extra(tensor);

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> host_data(nbytes);

    if (extra && extra->tensor) {
        ggml_ttnn_download_tensor(tensor, *extra->tensor, host_data.data());
    } else {
        std::fill(host_data.begin(), host_data.end(), 0);
    }

    std::memset(host_data.data() + offset, value, size);

    ggml_backend_ttnn_device_context * dev_ctx = (ggml_backend_ttnn_device_context *) buffer->buft->device->context;
    GGML_ASSERT(dev_ctx->device != nullptr);

    ttnn::Tensor device_tensor =
        ggml_ttnn_upload_tensor(tensor, host_data.data(), nbytes, dev_ctx->device);

    if (extra) {
        extra->tensor = std::make_unique<ttnn::Tensor>(std::move(device_tensor));
    }
}

static void ggml_backend_ttnn_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {
    if (ggml_is_quantized(tensor->type)) {
        GGML_ABORT("TTNN backend does not support quantized tensors");
    }

    ggml_backend_ttnn_device_context * dev_ctx = (ggml_backend_ttnn_device_context *) buffer->buft->device->context;
    GGML_ASSERT(dev_ctx->device != nullptr);

    const size_t nbytes = ggml_nbytes(tensor);
    ggml_ttnn_tensor * extra = ggml_ttnn_get_extra(tensor);

    if (offset == 0 && size == nbytes) {
        ttnn::Tensor device_tensor =
            ggml_ttnn_upload_tensor(tensor, data, nbytes, dev_ctx->device);
        if (extra) {
            extra->tensor = std::make_unique<ttnn::Tensor>(std::move(device_tensor));
        }
        return;
    }

    std::vector<uint8_t> host_data(nbytes);
    if (extra && extra->tensor) {
        ggml_ttnn_download_tensor(tensor, *extra->tensor, host_data.data());
    } else {
        std::fill(host_data.begin(), host_data.end(), 0);
    }

    std::memcpy(host_data.data() + offset, data, size);

    ttnn::Tensor device_tensor =
        ggml_ttnn_upload_tensor(tensor, host_data.data(), nbytes, dev_ctx->device);
    if (extra) {
        extra->tensor = std::make_unique<ttnn::Tensor>(std::move(device_tensor));
    }
}

static void ggml_backend_ttnn_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {
    ggml_ttnn_tensor * extra = ggml_ttnn_get_extra(tensor);
    if (!extra || !extra->tensor) {
        GGML_ABORT("TTNN backend tensor data not initialized");
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (offset == 0 && size == nbytes) {
        ggml_ttnn_download_tensor(tensor, *extra->tensor, data);
        return;
    }

    std::vector<uint8_t> host_data(nbytes);
    ggml_ttnn_download_tensor(tensor, *extra->tensor, host_data.data());
    std::memcpy(data, host_data.data() + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_ttnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    for (ggml_tensor * tensor : ctx->tensors) {
        ggml_backend_ttnn_buffer_memset_tensor(buffer, tensor, value, 0, ggml_nbytes(tensor));
    }
}

static const ggml_backend_buffer_i ggml_backend_ttnn_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_ttnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_ttnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_ttnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_ttnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_ttnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_ttnn_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_ttnn_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * ggml_backend_ttnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_TTNN_NAME;
}

static ggml_backend_buffer_t ggml_backend_ttnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_ttnn_buffer_context * ctx = new ggml_backend_ttnn_buffer_context();
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t alloc_size = size > 0 ? size : alignment;
    size_t buffer_size = size > 0 ? size : alloc_size;

    ctx->host_ptr = ggml_aligned_malloc(alloc_size);
    ctx->host_size = alloc_size;
    if (ctx->host_ptr == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_ttnn_buffer_interface, ctx, buffer_size);
}

static size_t ggml_backend_ttnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static size_t ggml_backend_ttnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_ttnn_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ggml_backend_ttnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_ttnn_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_ttnn_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_ttnn_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_ttnn_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_ttnn_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_ttnn_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_ttnn_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_ttnn_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static ggml_backend_buffer_type ttnn_buffer_type = {};
    static bool initialized = false;

    if (device != 0) {
        GGML_ABORT("TTNN backend supports only device 0");
    }

    if (!initialized) {
        ttnn_buffer_type = {
            /* .iface   = */ ggml_backend_ttnn_buffer_type_interface,
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_ttnn_reg(), device),
            /* .context = */ nullptr,
        };
        initialized = true;
    }

    return &ttnn_buffer_type;
}

//
// backend
//

static const char * ggml_backend_ttnn_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_TTNN_NAME;
}

static void ggml_backend_ttnn_free(ggml_backend_t backend) {
    ggml_backend_ttnn_context * ctx = (ggml_backend_ttnn_context *) backend->context;
    if (ctx && ctx->device) {
        ttnn::synchronize(ctx->device);
        ttnn::close_device(ctx->device);
        ctx->device = nullptr;
    }

    if (backend->device && backend->device->context) {
        ggml_backend_ttnn_device_context * dev_ctx =
            (ggml_backend_ttnn_device_context *) backend->device->context;
        dev_ctx->device = nullptr;
    }

    delete ctx;
}

static void ggml_backend_ttnn_synchronize(ggml_backend_t backend) {
    ggml_backend_ttnn_context * ctx = (ggml_backend_ttnn_context *) backend->context;
    if (ctx && ctx->device) {
        ttnn::synchronize(ctx->device);
    }
}

static enum ggml_status ggml_backend_ttnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_ttnn_context * ctx = (ggml_backend_ttnn_context *) backend->context;
    GGML_ASSERT(ctx && ctx->device);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_NONE) {
            continue;
        }

        ggml_ttnn_tensor * dst_extra = ggml_ttnn_get_extra(node);
        GGML_ASSERT(dst_extra);

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                auto & b = *ggml_ttnn_get_extra(node->src[1])->tensor;
                auto out = ttnn::matmul(a, b);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            case GGML_OP_ADD: {
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                auto & b = *ggml_ttnn_get_extra(node->src[1])->tensor;
                auto out = ttnn::add(a, b);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            case GGML_OP_GELU: {
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                auto out = ttnn::gelu(a);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            case GGML_OP_SOFT_MAX: {
                if (node->src[2] != nullptr) {
                    return GGML_STATUS_FAILED;
                }
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                auto out = ttnn::softmax(a, -1);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            case GGML_OP_RMS_NORM: {
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                float eps = 0.0f;
                std::memcpy(&eps, node->op_params, sizeof(float));
                auto out = ttnn::rms_norm(a, eps);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            case GGML_OP_GET_ROWS: {
                auto & a = *ggml_ttnn_get_extra(node->src[0])->tensor;
                auto & b = *ggml_ttnn_get_extra(node->src[1])->tensor;
                auto out = ttnn::embedding(b, a);
                dst_extra->tensor = std::make_unique<ttnn::Tensor>(std::move(out));
            } break;
            default:
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_ttnn_interface = {
    /* .get_name                = */ ggml_backend_ttnn_name,
    /* .free                    = */ ggml_backend_ttnn_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_ttnn_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_ttnn_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

//
// backend device
//

static const char * ggml_backend_ttnn_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_ttnn_device_context * ctx = (ggml_backend_ttnn_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_ttnn_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_ttnn_device_context * ctx = (ggml_backend_ttnn_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_ttnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_ttnn_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ttnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ttnn_device_get_name(dev);
    props->description = ggml_backend_ttnn_device_get_description(dev);
    props->type        = ggml_backend_ttnn_device_get_type(dev);
    ggml_backend_ttnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_ttnn_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_ttnn_device_context * ctx = (ggml_backend_ttnn_device_context *) dev->context;
    return ggml_backend_ttnn_init(ctx->device_id);
}

static bool ggml_backend_ttnn_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (ggml_is_quantized(op->type)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = op->src[i];
        if (!src) {
            continue;
        }
        if (ggml_is_quantized(src->type)) {
            return false;
        }
        if (src->view_src != nullptr || !ggml_is_contiguous(src)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_GELU:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GET_ROWS:
            return true;
        case GGML_OP_SOFT_MAX:
            return op->src[2] == nullptr;
        default:
            return false;
    }
}

static bool ggml_backend_ttnn_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_ttnn_buffer_type_get_name &&
        buft->device == dev;
}

static ggml_backend_buffer_type_t ggml_backend_ttnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_ttnn_device_context * ctx = (ggml_backend_ttnn_device_context *) dev->context;
    return ggml_backend_ttnn_buffer_type(ctx->device_id);
}

static const ggml_backend_device_i ggml_backend_ttnn_device_interface = {
    /* .get_name             = */ ggml_backend_ttnn_device_get_name,
    /* .get_description      = */ ggml_backend_ttnn_device_get_description,
    /* .get_memory           = */ ggml_backend_ttnn_device_get_memory,
    /* .get_type             = */ ggml_backend_ttnn_device_get_type,
    /* .get_props            = */ ggml_backend_ttnn_device_get_props,
    /* .init_backend         = */ ggml_backend_ttnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_ttnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_ttnn_supports_op,
    /* .supports_buft        = */ ggml_backend_ttnn_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// backend registry
//

struct ggml_backend_ttnn_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_ttnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_TTNN_NAME;
}

static size_t ggml_backend_ttnn_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_ttnn_reg_context * ctx = (ggml_backend_ttnn_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_ttnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_ttnn_reg_context * ctx = (ggml_backend_ttnn_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_ttnn_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_ttnn_reg_interface = {
    /* .get_name         = */ ggml_backend_ttnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_ttnn_reg_device_count,
    /* .get_device       = */ ggml_backend_ttnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_ttnn_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_ttnn_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_ttnn_reg_context * ctx = new ggml_backend_ttnn_reg_context();

            const int32_t device_count = 1;
            for (int32_t i = 0; i < device_count; ++i) {
                auto * dev_ctx = new ggml_backend_ttnn_device_context();
                dev_ctx->device_id = i;
                dev_ctx->name = GGML_TTNN_NAME + std::to_string(i);
                dev_ctx->description = "Tenstorrent TTNN device";
                dev_ctx->device = nullptr;

                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_ttnn_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx,
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_ttnn_reg_interface,
                /* .context     = */ ctx,
            };
        }
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_ttnn_init(int32_t device) {
    ggml_ttnn_enable_program_cache();

    ggml_backend_ttnn_device_context * dev_ctx =
        (ggml_backend_ttnn_device_context *) ggml_backend_reg_dev_get(ggml_backend_ttnn_reg(), device)->context;

    if (!dev_ctx) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    if (dev_ctx->device == nullptr) {
        dev_ctx->device = ttnn::open_device(device);
    }

    ggml_backend_ttnn_context * ctx = new ggml_backend_ttnn_context();
    ctx->device = dev_ctx->device;
    ctx->device_id = device;

    return new ggml_backend{
        /* .guid      = */ ggml_backend_ttnn_guid(),
        /* .interface = */ ggml_backend_ttnn_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_ttnn_reg(), device),
        /* .context   = */ ctx,
    };
}

bool ggml_backend_is_ttnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_ttnn_guid());
}

int32_t ggml_backend_ttnn_get_device(ggml_backend_t backend) {
    if (!ggml_backend_is_ttnn(backend)) {
        return -1;
    }
    ggml_backend_ttnn_context * ctx = (ggml_backend_ttnn_context *) backend->context;
    return ctx ? ctx->device_id : -1;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ttnn_reg)
