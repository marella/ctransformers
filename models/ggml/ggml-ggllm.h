// https://github.com/cmp-nct/ggllm.cpp/blob/master/ggml.h

GGML_API struct ggml_tensor* ggml_repeat2(struct ggml_context* ctx,
                                          struct ggml_tensor* a,
                                          struct ggml_tensor* b);
