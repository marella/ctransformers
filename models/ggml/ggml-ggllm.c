// https://github.com/cmp-nct/ggllm.cpp/blob/master/ggml.c

// ggml_repeat2

struct ggml_tensor *ggml_repeat2(struct ggml_context *ctx,
                                 struct ggml_tensor *a, struct ggml_tensor *b) {
  GGML_ASSERT(ggml_can_repeat(a, b));

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  if (ggml_are_same_shape(a, b) && !is_node) {
    return a;
  }

  struct ggml_tensor *result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

  result->op = GGML_OP_REPEAT2;
  result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// ggml_compute_forward_repeat2

static void ggml_compute_forward_repeat2_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    struct ggml_tensor *dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_can_repeat(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne0 / ne00);
  const int nr1 = (int)(ne1 / ne01);
  const int nr2 = (int)(ne2 / ne02);
  const int nr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  int i2k2 = 0;

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne03; k3++, i2k2 = 0) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne02; k2++, i2k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne01; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_vec_cpy_f32(
                    ne00,
                    (float *)((char *)dst->data + (i3 * ne03 + k3) * nb3 +
                              (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 +
                              (i0 * ne00) * nb0),
                    (float *)((char *)src0->data + (k3)*nb03 +
                              (i2k2 / nr2) * nb02 + (k1)*nb01));
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat2(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    struct ggml_tensor *dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_repeat2_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}
