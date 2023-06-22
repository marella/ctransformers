// Defines fileno on msys:
// inference&model based on the ggml falcon example PR from
// https://github.com/KerfuffleV2/ggml-falcon
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "ggml.h"
#include "libfalcon.h"
#include "llama-util.h"
#ifdef GGML_USE_CUBLAS
#include <cuda_runtime.h>

#include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifndef QK_K
#define QK_K 256
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
// disable "possible loss of data"
#pragma warning(disable : 4244 4267)
#endif

#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16

// available falcon models
enum falcon_e_model {
  FALCON_UNKNOWN,
  FALCON_7B,
  FALCON_40B,
};

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

typedef void (*offload_func_t)(struct ggml_tensor *tensor);

static const std::map<falcon_e_model, size_t> &FALCON_MEM_REQ_SCRATCH0() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 512ull * MB},
      {FALCON_40B, 1024ull * MB},
  };
  return k_sizes;
}

static const std::map<falcon_e_model, size_t> &FALCON_MEM_REQ_SCRATCH1() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 512ull * MB},
      {FALCON_40B, 1024ull * MB},
  };
  return k_sizes;
}

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<falcon_e_model, size_t> &FALCON_MEM_REQ_EVAL() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 768ull * MB},
      {FALCON_40B, 1536ull * MB},
  };
  return k_sizes;
}

// default hparams (Falcon 7B)
struct falcon_hparams {
  int32_t n_vocab = 65024;
  int32_t n_ctx = 2048;
  int32_t n_embd = 4544;
  int32_t n_head = 71;
  int32_t n_head_kv = 1;
  int32_t n_layer = 32;
  int32_t n_falcon_type = 7;  // 7 for Falcon-7B, 40 for Falcon-40B
  enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;

  bool operator!=(const falcon_hparams &other) const {
    return static_cast<bool>(memcmp(this, &other, sizeof(falcon_hparams)));
  }
};

static size_t FALCON_MEM_REQ_KV_SELF(const falcon_hparams &hparams,
                                     ggml_type wtype, int32_t n_ctx) {
  const int n_head_kv = hparams.n_head_kv;
  const int head_dim = hparams.n_embd / hparams.n_head;
  const int n_layer = hparams.n_layer;

  const int64_t ne = n_head_kv * head_dim * n_layer * n_ctx;

  return 2u * (ggml_tensor_overhead() + ne * ggml_type_size(wtype));
}

struct falcon_layer {
  // normalization
  struct ggml_tensor *input_layernorm;
  struct ggml_tensor *input_layernorm_b;
  struct ggml_tensor *attention_norm;    // Falcon-40B only
  struct ggml_tensor *attention_norm_b;  // Falcon-40B only

  // attention
  struct ggml_tensor *query_key_value;
  struct ggml_tensor *wo;

  // ff
  struct ggml_tensor *ffn_up;
  struct ggml_tensor *ffn_down;
};

struct falcon_kv_cache {
  struct ggml_tensor *k;
  struct ggml_tensor *v;

  struct ggml_context *ctx = NULL;

  llama_ctx_buffer buf;

  int n;  // number of tokens currently in the cache

  ~falcon_kv_cache() {
    if (ctx) {
      ggml_free(ctx);
    }
  }
};

struct falcon_model {
  falcon_e_model type = FALCON_UNKNOWN;

  falcon_hparams hparams;

  struct ggml_tensor *tok_embeddings;
  struct ggml_tensor *output_norm;
  struct ggml_tensor *output_norm_b;
  struct ggml_tensor *lm_head;
  // struct ggml_tensor* output;

  std::vector<falcon_layer> layers;

  int n_gpu_layers;
  int i_gpu_start;
  int i_gpu_last;

  // context
  struct ggml_context *ctx = NULL;
  std::map<std::string, struct ggml_tensor *> tensors;

  // key + value cache for the self attention
  // TODO: move to llama_state
  struct falcon_kv_cache kv_self;

  // the model memory buffer
  llama_ctx_buffer buf;

  // model memory mapped file
  std::unique_ptr<llama_mmap> mapping;

  // objects representing data potentially being locked in memory
  llama_mlock mlock_buf;
  llama_mlock mlock_mmap;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

  ~falcon_model() {
    if (ctx) {
      ggml_free(ctx);
    }

#ifdef GGML_USE_CUBLAS
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cuda_free_data(tensors_by_name[i].second);
    }
#elif defined(GGML_USE_CLBLAST)
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cl_free_data(tensors_by_name[i].second);
    }
#endif
  }
};

struct falcon_context {
  std::mt19937 rng;

  int64_t t_load_us = 0;
  int64_t t_start_us = 0;
  bool has_evaluated_once = false;

  int64_t t_sample_us = 0;
  int64_t t_eval_us = 0;
  int64_t t_p_eval_us = 0;

  int32_t n_sample = 0;  // number of tokens sampled
  int32_t n_eval = 0;    // number of eval calls
  int32_t n_p_eval =
      0;  // number of tokens in eval calls for the prompt (with batch size > 1)

  falcon_model model;
  llama_vocab vocab;

  size_t mem_per_token = 0;

  // decode output (2-dimensional array: [n_tokens][n_vocab])
  std::vector<float> logits;
  bool logits_all = false;

  // input embedding (1-dimensional array: [n_embd])
  std::vector<float> embedding;

  // memory buffers used to evaluate the model
  // TODO: move in llama_state
  llama_ctx_buffer buf_compute;
  llama_ctx_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

#ifdef GGML_USE_METAL
  ggml_metal_context *ctx_metal = NULL;
#endif

  int buf_last = 0;
  size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = {0};

  void use_buf(struct ggml_context *ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
    size_t last_size = 0;

    if (i == -1) {
      last_size = ggml_set_scratch(ctx, {
                                            0,
                                            0,
                                            nullptr,
                                        });
    } else {
      auto &buf = buf_scratch[i];
      last_size = ggml_set_scratch(ctx, {
                                            0,
                                            buf.size,
                                            buf.addr,
                                        });
    }

    if (buf_last >= 0) {
      buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
    }

    buf_last = i;
#else
    (void)i;
    (void)ctx;
#endif
  }

  size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
    return buf_max_size[i];
#else
    (void)i;
    return 0;
#endif
  }
};

struct falcon_file_loader {
  llama_file file;
  llama_file_version file_version;
  falcon_hparams hparams;
  llama_vocab vocab;

  falcon_file_loader(const char *fname, size_t file_idx,
                     llama_load_tensors_map &tensors_map)
      : file(fname, "rb") {
    read_magic();
    if (file_version <= 1)
      fprintf(stderr,
              "falcon.cpp: file version %d - Use falcon_quantize to update "
              "version and improve performance\n",
              file_version);
    read_hparams();
    read_vocab();
    read_tensor_metadata(file_idx, tensors_map);
  }
  void read_magic() {
    uint32_t magic = file.read_u32();

    if (magic == LLAMA_FILE_MAGIC_GGML) {
      file_version = LLAMA_FILE_VERSION_GGML;
      return;
    }

    uint32_t version = file.read_u32();

    switch (magic) {
      case LLAMA_FILE_MAGIC_GGMF:
        switch (version) {
          case 1:
            file_version = LLAMA_FILE_VERSION_GGMF_V1;
            return;
        }
        break;
      case LLAMA_FILE_MAGIC_GGJT:
        switch (version) {
          case 1:
            file_version = LLAMA_FILE_VERSION_GGJT_V1;
            return;
          case 2:
            file_version = LLAMA_FILE_VERSION_GGJT_V2;
            return;
          case 3:
            file_version = LLAMA_FILE_VERSION_GGJT_V3;
            return;
        }
    }

    throw std::runtime_error(
        format("unknown (magic, version) combination: %08x, %08x; is this "
               "really a GGML file?",
               magic, version));
  }
  void read_hparams() {
    hparams.n_vocab = file.read_u32();
    hparams.n_embd = file.read_u32();
    hparams.n_head = file.read_u32();
    hparams.n_head_kv = file.read_u32();
    hparams.n_layer = file.read_u32();
    hparams.n_falcon_type = file.read_u32();
    if (file_version == LLAMA_FILE_VERSION_GGML) {
      int32_t ftype = file.read_u32();
      const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;
      hparams.ftype =
          (enum llama_ftype)(hparams.ftype % GGML_QNT_VERSION_FACTOR);
    } else {
      hparams.ftype = (enum llama_ftype)file.read_u32();
    }

    // hparams.ftype %= GGML_QNT_VERSION_FACTOR;
  }
  void read_vocab() {
    vocab.id_to_token.resize(hparams.n_vocab);

    for (uint32_t i = 0; i < (uint32_t)hparams.n_vocab; i++) {
      uint32_t len = file.read_u32();
      std::string word = file.read_string(len);

      float score = 0.0f;  // flacon does not have scores in vocab, scores are a
                           // sentencepiece addition
      if (file_version >= LLAMA_FILE_VERSION_GGMF_V1) {
        file.read_raw(&score, sizeof(score));
      }

      vocab.token_to_id[word] = i;

      auto &tok_score = vocab.id_to_token[i];
      tok_score.tok = std::move(word);
      tok_score.score = score;
    }
  }
  void read_tensor_metadata(size_t file_idx,
                            llama_load_tensors_map &tensors_map) {
    while (file.tell() < file.size) {
      llama_load_tensor_shard shard;
      uint32_t n_dims = file.read_u32();
      uint32_t name_len = file.read_u32();
      shard.type = (enum ggml_type)file.read_u32();
      shard.ne.resize(n_dims);
      file.read_raw(shard.ne.data(), sizeof(shard.ne[0]) * n_dims);
      std::string name = file.read_string(name_len);
      if (n_dims < 1 || n_dims > 2) {
        throw std::runtime_error(
            format("falcon.cpp: tensor '%s' should not be %u-dimensional",
                   name.c_str(), n_dims));
      }
      switch (shard.type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
          break;
        default: {
          throw std::runtime_error(
              format("unrecognized tensor type %u\n", shard.type));
        }
      }

      if (file_version >= LLAMA_FILE_VERSION_GGJT_V1) {
        // skip to the next multiple of 32 bytes
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
      }
      shard.file_idx = file_idx;
      shard.file_off = file.tell();

      shard.calc_size();
      file.seek(shard.size, SEEK_CUR);

      auto it = tensors_map.name_to_idx.find(name);
      size_t idx;
      if (it != tensors_map.name_to_idx.end()) {
        idx = it->second;
      } else {
        tensors_map.tensors.emplace_back(name);
        idx = tensors_map.tensors.size() - 1;
        tensors_map.name_to_idx.emplace(name, idx);
      }
      tensors_map.tensors.at(idx).shards.push_back(shard);
    }
  }
};

struct falcon_file_saver {
  llama_file file;
  falcon_file_loader *any_file_loader;
  falcon_file_saver(const char *fname, falcon_file_loader *any_file_loader,
                    enum llama_ftype new_ftype)
      : file(fname, "wb"), any_file_loader(any_file_loader) {
    fprintf(stderr, "falcon.cpp: saving model to %s\n", fname);
    write_magic();
    write_hparams(new_ftype);
    write_vocab();
  }
  void write_magic() {
    file.write_u32(LLAMA_FILE_MAGIC);    // magic
    file.write_u32(LLAMA_FILE_VERSION);  // version
  }
  void write_hparams(enum llama_ftype new_ftype) {
    const falcon_hparams &hparams = any_file_loader->hparams;
    file.write_u32(hparams.n_vocab);
    file.write_u32(hparams.n_embd);
    file.write_u32(hparams.n_head);
    file.write_u32(hparams.n_head_kv);
    file.write_u32(hparams.n_layer);
    file.write_u32(hparams.n_falcon_type);
    file.write_u32(new_ftype);
  }
  void write_vocab() {
    if (any_file_loader->file_version == LLAMA_FILE_VERSION_GGML) {
      fprintf(stderr,
              "falcon.cpp: WARNING: input is an old file that doesn't have "
              "scores; will add dummy scores\n");
    }
    uint32_t n_vocab = any_file_loader->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
      const auto &token_score = any_file_loader->vocab.id_to_token.at(i);
      file.write_u32((uint32_t)token_score.tok.size());
      file.write_raw(token_score.tok.data(), token_score.tok.size());
      file.write_raw(&token_score.score, sizeof(token_score.score));
    }
  }
  void write_tensor(llama_load_tensor &tensor, enum ggml_type new_type,
                    const void *new_data, size_t new_size) {
    switch (new_type) {
      case GGML_TYPE_F32:
      case GGML_TYPE_F16:
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q5_0:
      case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q8_0:
      case GGML_TYPE_Q2_K:
      case GGML_TYPE_Q3_K:
      case GGML_TYPE_Q4_K:
      case GGML_TYPE_Q5_K:
      case GGML_TYPE_Q6_K:
        break;
      default:
        LLAMA_ASSERT(false);
    }
    file.write_u32((uint32_t)tensor.ne.size());
    file.write_u32((uint32_t)tensor.name.size());
    file.write_u32(new_type);
    file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
    file.write_raw(tensor.name.data(), tensor.name.size());
    file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
    LLAMA_ASSERT(new_size == llama_calc_tensor_size(tensor.ne, new_type));
    file.write_raw(new_data, new_size);
  }
};

struct falcon_model_loader {
  std::vector<std::unique_ptr<falcon_file_loader>> file_loaders;
  llama_load_tensors_map tensors_map;
  bool use_mmap;
  size_t num_ggml_tensors_created = 0;
  struct ggml_context *ggml_ctx = NULL;
  std::unique_ptr<llama_mmap> mapping;

  falcon_model_loader(const std::string &fname_base, bool use_mmap,
                      bool vocab_only) {
    auto *first_file =
        new falcon_file_loader(fname_base.c_str(), 0, tensors_map);
    file_loaders.emplace_back(first_file);
    uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
    for (uint32_t i = 1; i < n_parts; i++) {
      std::string fname = fname_base + "." + std::to_string(i);
      auto *ith_file = new falcon_file_loader(fname.c_str(), i, tensors_map);
      file_loaders.emplace_back(ith_file);
      if (ith_file->hparams != first_file->hparams) {
        throw std::runtime_error(
            format("falcon.cpp: hparams inconsistent between files"));
      }
    }
    if (!llama_mmap::SUPPORTED) {
      use_mmap = false;
    }
    if (use_mmap && alignment_prevents_mmap()) {
      fprintf(stderr,
              "falcon.cpp: can't use mmap because tensors are not aligned; "
              "convert to new format to avoid this\n");
      use_mmap = false;
    }
    this->use_mmap = use_mmap;
    for (llama_load_tensor &lt : tensors_map.tensors) {
      lt.calc_all();
    }
  }

  bool alignment_prevents_mmap() {
    for (const llama_load_tensor &lt : tensors_map.tensors) {
      for (const llama_load_tensor_shard &shard : lt.shards) {
        if (shard.file_off & 3) {
          return true;
        }
      }
    }
    return false;
  }

  uint32_t guess_n_parts() const {
    auto it =
        tensors_map.name_to_idx.find("transformer.word_embeddings.weight");
    if (it == tensors_map.name_to_idx.end()) {
      throw std::runtime_error(std::string("missing tok_embeddings.weight"));
    }
    const llama_load_tensor &lt = tensors_map.tensors.at(it->second);
    return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
  }

  void calc_sizes(size_t *ctx_size_p, size_t *mmapped_size_p) const {
    *ctx_size_p = *mmapped_size_p = 0;
    for (const llama_load_tensor &lt : tensors_map.tensors) {
      *ctx_size_p += ggml_tensor_overhead();
      *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
    }
  }

  struct ggml_tensor *get_tensor(const std::string &name,
                                 const std::vector<uint32_t> &ne,
                                 ggml_backend backend) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      throw std::runtime_error(std::runtime_error(format(
          "falcon.cpp: tensor '%s' is missing from model", name.c_str())));
    }
    llama_load_tensor &lt = tensors_map.tensors.at(it->second);
    if (lt.ne != ne) {
      throw std::runtime_error(
          format("falcon.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                 name.c_str(), llama_format_tensor_shape(ne).c_str(),
                 llama_format_tensor_shape(lt.ne).c_str()));
    }

    return get_tensor_for(lt, backend);
  }

  struct ggml_tensor *get_tensor_for(llama_load_tensor &lt,
                                     ggml_backend backend) {
    struct ggml_tensor *tensor;
    if (backend != GGML_BACKEND_CPU) {
      ggml_set_no_alloc(ggml_ctx, true);
    }
    if (lt.ne.size() == 2) {
      tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
    } else {
      LLAMA_ASSERT(lt.ne.size() == 1);
      tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
    }
    ggml_set_name(tensor, lt.name.c_str());
    LLAMA_ASSERT(
        lt.ggml_tensor ==
        NULL);  // if this fails, we called get_tensor twice on the same tensor

    if (backend != GGML_BACKEND_CPU) {
      ggml_set_no_alloc(ggml_ctx, use_mmap);
    }
    tensor->backend = backend;
    lt.ggml_tensor = tensor;
    num_ggml_tensors_created++;
    return tensor;
  }

  void done_getting_tensors() const {
    if (num_ggml_tensors_created != tensors_map.tensors.size()) {
      throw std::runtime_error(
          std::string("falcon.cpp: file contained more tensors than expected"));
    }
  }

  void load_all_data(llama_progress_callback progress_callback,
                     void *progress_callback_user_data, llama_mlock *lmlock) {
    size_t data_size = 0;
    size_t prefetch_size = 0;
    size_t lock_size = 0;
    for (const llama_load_tensor &lt : tensors_map.tensors) {
      data_size += lt.size;
      if (lt.ggml_tensor->backend == GGML_BACKEND_CPU) {
        prefetch_size += lt.size;
      }
    }

    if (use_mmap) {
      mapping.reset(new llama_mmap(&file_loaders.at(0)->file, prefetch_size));
      if (lmlock) {
        lmlock->init(mapping->addr);
      }
    }

    size_t done_size = 0;
    for (llama_load_tensor &lt : tensors_map.tensors) {
      if (progress_callback) {
        progress_callback((float)done_size / data_size,
                          progress_callback_user_data);
      }
      LLAMA_ASSERT(lt.ggml_tensor);  // unused tensors should have been caught
                                     // by load_data already
      lt.data = (uint8_t *)lt.ggml_tensor->data;

      // allocate temp buffer if not using mmap
      if (!use_mmap && lt.data == NULL) {
        GGML_ASSERT(lt.ggml_tensor->backend != GGML_BACKEND_CPU);
        lt.data = (uint8_t *)malloc(ggml_nbytes(lt.ggml_tensor));
      }

      load_data_for(lt);

      switch (lt.ggml_tensor->backend) {
        case GGML_BACKEND_CPU:
          lt.ggml_tensor->data = lt.data;
          if (use_mmap && lmlock) {
            lock_size += lt.size;
            lmlock->grow_to(lock_size);
          }
          break;
#if defined(GGML_USE_CUBLAS)
        case GGML_BACKEND_GPU:
        case GGML_BACKEND_GPU_SPLIT:
          ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);
          if (!use_mmap) {
            free(lt.data);
          }
          break;
#elif defined(GGML_USE_CLBLAST)
        case GGML_BACKEND_GPU:
          ggml_cl_transform_tensor(lt.data, lt.ggml_tensor);
          if (!use_mmap) {
            free(lt.data);
          }
          break;
#endif
        default:
          continue;
      }

      done_size += lt.size;
    }
  }

  void load_data_for(llama_load_tensor &lt) {
    if (use_mmap) {
      LLAMA_ASSERT(lt.shards.size() == 1);
      lt.data = (uint8_t *)mapping->addr + lt.shards.at(0).file_off;
    } else if (lt.split_type == SPLIT_NONE) {
      llama_file &file = file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off, SEEK_SET);
      file.read_raw(lt.data, lt.size);
    } else if (lt.split_type == SPLIT_BY_ROWS) {
      size_t offset = 0;
      for (llama_load_tensor_shard &shard : lt.shards) {
        llama_file &file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        file.read_raw(lt.data + offset, shard.size);
        offset += shard.size;
      }
      LLAMA_ASSERT(offset == lt.size);
    } else if (lt.split_type == SPLIT_BY_COLUMNS) {
      // Let's load the data into temporary buffers to ensure the OS performs
      // large loads.
      std::vector<llama_buffer> tmp_bufs(lt.shards.size());
      for (size_t i = 0; i < lt.shards.size(); i++) {
        llama_load_tensor_shard &shard = lt.shards.at(i);
        llama_file &file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        tmp_bufs.at(i).resize(shard.size);
        file.read_raw(tmp_bufs.at(i).addr, shard.size);
      }
      // Then reshape.
      size_t num_rows = lt.ne.at(1);
      size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
      size_t out_offset = 0;
      for (size_t row = 0; row < num_rows; row++) {
        for (llama_buffer &tmp_buf : tmp_bufs) {
          memcpy(lt.data + out_offset, tmp_buf.addr + row * per_shard_row_size,
                 per_shard_row_size);
          out_offset += per_shard_row_size;
        }
      }
      LLAMA_ASSERT(out_offset == lt.size);
    }
    if (0) {
      print_checksum(lt);
    }
  }

  static void print_checksum(llama_load_tensor &lt) {
    uint32_t sum = 0;
    for (size_t i = 0; i < lt.size; i++) {
      uint8_t byte = lt.data[i];
      sum = byte + (sum << 6) + (sum << 16) - sum;  // sdbm hash
    }
    fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
            llama_format_tensor_shape(lt.ne).c_str(), lt.size);
  }
};

//
// kv cache
//

static bool kv_cache_init(const struct falcon_hparams &hparams,
                          struct falcon_kv_cache &cache, ggml_type wtype,
                          int n_ctx, int n_gpu_layers) {
  const int64_t n_layer = hparams.n_layer;
  const int64_t head_dim = hparams.n_embd / hparams.n_head;
  const int64_t n_elements =
      hparams.n_layer * n_ctx * head_dim * hparams.n_head_kv;

  cache.buf.resize(FALCON_MEM_REQ_KV_SELF(hparams, wtype, n_ctx));

  struct ggml_init_params params;
  params.mem_size = cache.buf.size;
  params.mem_buffer = cache.buf.addr;
  params.no_alloc = false;

  cache.ctx = ggml_init(params);

  if (!cache.ctx) {
    fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
    return false;
  }

  cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
  cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
  ggml_set_name(cache.k, "cache_k");
  ggml_set_name(cache.v, "cache_v");

  (void)n_gpu_layers;
#ifdef GGML_USE_CUBLAS
  if (n_gpu_layers > n_layer + 1) {
    ggml_cuda_assign_buffers_no_scratch(cache.k);
    ggml_cuda_assign_buffers_no_scratch(cache.v);
  }
#endif  // GGML_USE_CUBLAS

  return true;
}

struct falcon_context_params falcon_context_default_params() {
  struct falcon_context_params result = {
      /*.n_ctx                       =*/512,
      /*.n_batch                     =*/512,
      /*.n_gpu_layers                  =*/0,
      /*.i_gpu_start                 =*/-1,
      /*.i_gpu_last                   =*/-1,
      /*.main_gpu                    =*/0,
      /*.tensor_split                =*/{0},
      /*.seed                        =*/-1,
      /*.f16_kv                      =*/false,
      /*.logits_all                  =*/false,
      /*.vocab_only                  =*/false,
      /*.use_mmap                    =*/true,
      /*.use_mlock                   =*/false,
      /*.embedding                   =*/false,
      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr,
  };

  return result;
}

struct falcon_model_quantize_params falcon_model_quantize_default_params() {
  struct falcon_model_quantize_params result = {
      /*.nthread                     =*/0,
      /*.ftype                       =*/LLAMA_FTYPE_MOSTLY_Q5_1,
      /*.allow_requantize            =*/false,
      /*.quantize_output_tensor      =*/true,
  };

  return result;
}

//
// model loading
//

static const char *falcon_model_type_name(falcon_e_model type) {
  switch (type) {
    case FALCON_7B:
      return "7B";
    case FALCON_40B:
      return "40B";
    default:
      LLAMA_ASSERT(false);
  }
}

// dynamically gets all tensors from a layer
std::vector<ggml_tensor *> get_tensors_from_layer(falcon_layer &layer) {
  std::vector<ggml_tensor *> tensors;
  ggml_tensor **tensor_ptr = reinterpret_cast<ggml_tensor **>(
      &layer);  // Cast to the pointer to ggml_tensor pointer

  // Iterate through the members and store their addresses in the vector
  for (std::size_t i = 0; i < sizeof(falcon_layer) / sizeof(ggml_tensor *);
       ++i) {
    tensors.push_back(tensor_ptr[i]);
  }

  return tensors;
}
// get vram size of all tensors in a layer (todo: split handling)
size_t calculate_layer_vram_bytes(const falcon_layer &layer) {
  size_t size = 0;
  auto tensors = get_tensors_from_layer(const_cast<falcon_layer &>(layer));

  // Add the size of each member with GPU backend
  for (const auto &tensor : tensors) {
    if (tensor != nullptr && tensor->backend != GGML_BACKEND_CPU) {
      size += ggml_nbytes(tensor);
    }
  }

  return size;
}

static void falcon_model_load_internal(
    const std::string &fname, falcon_context &lctx, int n_ctx, int n_batch,
    int n_gpu_layers, int main_gpu, const float *tensor_split,
    ggml_type memory_type, bool use_mmap, bool use_mlock, bool vocab_only,
    llama_progress_callback progress_callback,
    void *progress_callback_user_data) {
  lctx.t_start_us = ggml_time_us();

  std::unique_ptr<falcon_model_loader> ml(
      new falcon_model_loader(fname, use_mmap, vocab_only));

  lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
  auto &model = lctx.model;
  model.hparams = ml->file_loaders.at(0)->hparams;
  model.n_gpu_layers = n_gpu_layers;

  llama_file_version file_version = ml->file_loaders.at(0)->file_version;
  auto &hparams = model.hparams;

  {
    switch (hparams.n_layer) {
      case 32:
        model.type = falcon_e_model::FALCON_7B;
        break;
      case 60:
        model.type = falcon_e_model::FALCON_40B;
        break;
      default: {
        if (hparams.n_falcon_type == 7) {
          model.type = falcon_e_model::FALCON_7B;
        } else if (hparams.n_falcon_type == 40) {
          model.type = falcon_e_model::FALCON_40B;
        } else {
          LLAMA_ASSERT(false);
        }
      } break;
    }

    hparams.n_ctx = n_ctx;
  }

  const uint32_t n_ff = 4 * model.hparams.n_embd;

  if (file_version < LLAMA_FILE_VERSION_GGJT_V2) {
    if (hparams.ftype != LLAMA_FTYPE_ALL_F32 &&
        hparams.ftype != LLAMA_FTYPE_MOSTLY_F16 &&
        hparams.ftype != LLAMA_FTYPE_MOSTLY_Q8_0) {
      throw std::runtime_error(
          format("this format is no longer supported (see "
                 "https://github.com/ggerganov/llama.cpp/pull/1405)"));
    }
  }

  if (file_version < LLAMA_FILE_VERSION_GGJT_V3) {
    if (hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ||
        hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_1 ||
        hparams.ftype == LLAMA_FTYPE_MOSTLY_Q8_0) {
      throw std::runtime_error(
          format("this format is no longer supported (see "
                 "https://github.com/ggerganov/llama.cpp/pull/1508)"));
    }
  }

  if (vocab_only) {
    return;
  }

  auto &ctx = model.ctx;

  size_t ctx_size;
  size_t mmapped_size;
  ml->calc_sizes(&ctx_size, &mmapped_size);

  // create the ggml context
  {
    lctx.model.buf.resize(ctx_size);
    if (use_mlock) {
      lctx.model.mlock_buf.init(lctx.model.buf.addr);
      lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/lctx.model.buf.size,
        /*.mem_buffer =*/lctx.model.buf.addr,
        /*.no_alloc   =*/ml->use_mmap,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      throw std::runtime_error(format("ggml_init() failed"));
    }
  }

  (void)main_gpu;
#if defined(GGML_USE_CUBLAS)
  ggml_cuda_set_main_device(main_gpu);
#define LLAMA_BACKEND_OFFLOAD GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
  fprintf(stderr, "%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_BACKEND_OFFLOAD GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU
#else
#define LLAMA_BACKEND_OFFLOAD GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

  size_t vram_total = 0;
  size_t vram_free = 0;
  const size_t vram_reserved =
      512 *
      MB;  // that amount of VRAM is to stay free on GPU (headroom for other
           // processes - may be reduced in pure server environments)
  size_t vram_overhead =
      1250 *
      MB;  // this amount of vram is estimated for non weight storage buffers on
           // VRAM (no big difference between 7B and 40B, needs to increase when
           // more work is offloaded in the future)
#if defined(GGML_USE_CUBLAS)
  // cublas is used in 32 bit mode, temporary cuda storage/conversion buffers
  // are needed for batch ingestion ( could be run in 16 bit mode without
  // performance downgrade and save half the VRAM)
  if (model.type == FALCON_40B && n_batch > 1) {
    vram_overhead += (1024 + 288 + 256) * MB;
  }
  if (model.type == FALCON_7B && n_batch > 1) {
    vram_overhead += (315 + 80 + 78) * MB;
  }
  cudaMemGetInfo(
      &vram_free,
      &vram_total);  // this should go in ggml-cuda.cu but I don't want to make
                     // Johannes life harder by modifying that yet
#endif

  // prepare memory for the weights
  size_t vram_weights = 0;
  size_t vram_scratch = 0;

  (void)vram_scratch;
  (void)n_batch;
  // calculate scratch buffer size and allocate it
#ifdef GGML_USE_CUBLAS
  // vram_scratch = n_batch * MB;
  vram_scratch = 0;  // these are not used until we have multi operation support
  ggml_cuda_set_scratch_size(vram_scratch);
#endif  // GGML_USE_CUBLAS

  {
    const uint32_t n_embd = hparams.n_embd;
    const uint32_t n_head = hparams.n_head;
    const uint32_t n_head_kv = hparams.n_head_kv;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_ff = 4 * model.hparams.n_embd;
    const uint32_t n_vocab = hparams.n_vocab;
    const uint32_t head_dim = hparams.n_embd / hparams.n_head;

    ml->ggml_ctx = ctx;

    model.tok_embeddings = ml->get_tensor("transformer.word_embeddings.weight",
                                          {n_embd, n_vocab}, GGML_BACKEND_CPU);

    ggml_backend backend_norm;
    ggml_backend backend_output;
    // disabled norm/output offloading until further tests, causes silent crash
    // at the moment
    if (n_gpu_layers > int(n_layer) && false) {  // NOLINT
      backend_norm = LLAMA_BACKEND_OFFLOAD;
      backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
    } else {
      backend_norm = GGML_BACKEND_CPU;
      backend_output = GGML_BACKEND_CPU;
    }

    // "output" tensor
    {
      model.output_norm =
          ml->get_tensor("transformer.ln_f.weight", {n_embd}, backend_norm);
      model.output_norm_b =
          ml->get_tensor("transformer.ln_f.bias", {n_embd}, backend_norm);
      model.lm_head =
          ml->get_tensor("lm_head.weight", {n_embd, n_vocab}, backend_output);
    }

    if (backend_norm != GGML_BACKEND_CPU) {
      vram_weights +=
          ggml_nbytes(model.output_norm) + ggml_nbytes(model.output_norm_b);
      vram_free -=
          ggml_nbytes(model.output_norm) + ggml_nbytes(model.output_norm_b);
    }
    if (backend_output != GGML_BACKEND_CPU) {
      vram_weights += ggml_nbytes(model.lm_head);
      vram_free -= ggml_nbytes(model.lm_head);
    }

    int i_gpu_start = n_layer - n_gpu_layers;
    if (i_gpu_start < 0)
      i_gpu_start = 0;  // n_gpu_layers can be larger than n_layer

    int i_gpu_last = n_layer;  // allows to terminate the offloading earlier.
                               // TODO: instead do a proper calculation run and
                               // determine the start before the loop
    model.i_gpu_start = i_gpu_start;
    model.i_gpu_last = i_gpu_last;  // if VRAM doesn't run out i_gpu_last is
                                    // always the last layer

    model.layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
      const ggml_backend backend = (int(i) < i_gpu_start || int(i) > i_gpu_last)
                                       ? GGML_BACKEND_CPU
                                       : LLAMA_BACKEND_OFFLOAD;  // NOLINT
      const ggml_backend backend_split =
          (int(i) < i_gpu_start || int(i) > i_gpu_last)
              ? GGML_BACKEND_CPU
              : LLAMA_BACKEND_OFFLOAD_SPLIT;  // NOLINT

      auto &layer = model.layers[i];

      std::string layers_i = "layers." + std::to_string(i);
      std::string str_i = std::to_string(i);

      if (model.type == FALCON_40B) {
        layer.input_layernorm =
            ml->get_tensor("transformer.h." + str_i + ".ln_mlp.weight",
                           {n_embd}, GGML_BACKEND_CPU);
        layer.input_layernorm_b =
            ml->get_tensor("transformer.h." + str_i + ".ln_mlp.bias", {n_embd},
                           GGML_BACKEND_CPU);
        layer.attention_norm =
            ml->get_tensor("transformer.h." + str_i + ".ln_attn.weight",
                           {n_embd}, GGML_BACKEND_CPU);
        layer.attention_norm_b =
            ml->get_tensor("transformer.h." + str_i + ".ln_attn.bias", {n_embd},
                           GGML_BACKEND_CPU);
      } else  // FALCON_7B
      {
        layer.input_layernorm =
            ml->get_tensor("transformer.h." + str_i + ".input_layernorm.weight",
                           {n_embd}, backend);
        layer.input_layernorm_b =
            ml->get_tensor("transformer.h." + str_i + ".input_layernorm.bias",
                           {n_embd}, GGML_BACKEND_CPU);
      }

      layer.query_key_value = ml->get_tensor(
          "transformer.h." + str_i + ".self_attention.query_key_value.weight",
          {n_embd, (n_head_kv * 2 + n_head) * head_dim}, backend_split);
      layer.wo = ml->get_tensor(
          "transformer.h." + str_i + ".self_attention.dense.weight",
          {n_embd, n_embd}, backend_split);

      layer.ffn_up =
          ml->get_tensor("transformer.h." + str_i + ".mlp.dense_h_to_4h.weight",
                         {n_embd, n_ff}, backend_split);  // before gelu
      layer.ffn_down =
          ml->get_tensor("transformer.h." + str_i + ".mlp.dense_4h_to_h.weight",
                         {n_ff, n_embd}, backend_split);  // after gelu

      if (backend != GGML_BACKEND_CPU) {
        size_t vram_layer = 0;
        vram_layer = calculate_layer_vram_bytes(layer);
        vram_weights += vram_layer;
        vram_free =
            (vram_layer > vram_free)
                ? 0
                : vram_free -
                      vram_layer;  // simulate the layer being loaded in VRAM
        // test if we have enough VRAM to offload the next layer
        if (i < n_layer && vram_free <= (vram_overhead + vram_scratch +
                                         vram_reserved + vram_layer)) {
          fprintf(stderr,
                  "INFO: Not enough VRAM to load all requested layers - at "
                  "layer %d of %d: skipping\n",
                  i, n_layer);
          model.n_gpu_layers = n_gpu_layers;
          i_gpu_last = i;
          model.i_gpu_last = i_gpu_last;
          n_gpu_layers = i_gpu_last - i_gpu_start;
          printf("INFO: %d layers will be offloaded to GPU (layers %d to %d)\n",
                 n_gpu_layers, i_gpu_start + 1, i_gpu_last + 1);
        }
      }
    }
  }

  ml->done_getting_tensors();

  // print memory requirements
  {
    // this is the total memory required to run the inference
    // TODO: this calculation is still wrong
    int64_t mem_required = ctx_size + mmapped_size -
                           vram_weights +  // weights in VRAM not in memory
                           FALCON_MEM_REQ_SCRATCH0().at(model.type) +
                           FALCON_MEM_REQ_SCRATCH1().at(model.type) +
                           FALCON_MEM_REQ_EVAL().at(model.type);

    if (mem_required < 0) mem_required = 0;

    // this is the memory required by one llama_state
    const size_t mem_required_state =
        FALCON_MEM_REQ_KV_SELF(model.hparams, memory_type, n_ctx);

    // moved scratch allocation of vram to top
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));
#else
    (void)n_gpu_layers;
#endif
  }

  // populate `tensors_by_name`
  for (llama_load_tensor &lt : ml->tensors_map.tensors) {
    model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
  }

  (void)tensor_split;
#if defined(GGML_USE_CUBLAS)
  { ggml_cuda_set_tensor_split(tensor_split); }
#endif

  ml->load_all_data(progress_callback, progress_callback_user_data,
                    use_mlock ? &lctx.model.mlock_mmap : NULL);

  if (progress_callback) {
    progress_callback(1.0f, progress_callback_user_data);
  }

#if defined(GGML_USE_CUBLAS)
  // size_t vram_free_simulated = vram_free;
  cudaMemGetInfo(
      &vram_free,
      &vram_total);  // this should go in ggml-cuda.cu but I don't want to make
                     // Johannes life harder by modifying that yet
#endif

  model.mapping = std::move(ml->mapping);

  // loading time will be recalculate after the first eval, so
  // we take page faults deferred by mmap() into consideration
  lctx.t_load_us = ggml_time_us() - lctx.t_start_us;
}

static bool falcon_model_load(const std::string &fname, falcon_context &lctx,
                              int n_ctx, int n_batch, int n_gpu_layers,
                              int main_gpu, float *tensor_split,
                              ggml_type memory_type, bool use_mmap,
                              bool use_mlock, bool vocab_only,
                              llama_progress_callback progress_callback,
                              void *progress_callback_user_data) {
  try {
    falcon_model_load_internal(fname, lctx, n_ctx, n_batch, n_gpu_layers,
                               main_gpu, tensor_split, memory_type, use_mmap,
                               use_mlock, vocab_only, progress_callback,
                               progress_callback_user_data);
    return true;
  } catch (const std::exception &err) {
    fprintf(stderr, "error loading model: %s\n", err.what());
    return false;
  }
}

// evaluate the transformer
//
//   - lctx:         llama context
//   - tokens:       new batch of tokens to process
//   - n_past:       the context size so far
//   - n_threads:    number of threads to use
//   - cgraph_fname: filename of the exported computation graph
//
static bool falcon_eval_internal(falcon_context &lctx,
                                 const llama_token *tokens, const int n_tokens,
                                 const int n_past, const int n_threads,
                                 const char *cgraph_fname, int debug_timings) {
  const int64_t t_start_us = ggml_time_us();

  const int N = n_tokens;
  // const int N = embd_inp.size();

  const auto &model = lctx.model;
  const auto &hparams = model.hparams;

  const auto &kv_self = model.kv_self;

  LLAMA_ASSERT(!!kv_self.ctx);

  const int n_embd = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx = hparams.n_ctx;
  const int n_head = hparams.n_head;
  const int n_head_kv = hparams.n_head_kv;
  const int n_vocab = hparams.n_vocab;
  const int n_falcon_type = hparams.n_falcon_type;
  const int n_gpu_layers = model.n_gpu_layers;
  const size_t head_dim = n_embd / n_head;  // == n_rot in llama

  auto &mem_per_token = lctx.mem_per_token;
  auto &buf_compute = lctx.buf_compute;

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ggml_context *ctx0 = ggml_init(params);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are
  // degrading the performance
  ggml_cgraph gf = {};
  gf.n_threads =
      N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : n_threads;

  struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
  ggml_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ggml_element_size(embd));

  struct ggml_tensor *cur;
  struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);
  struct ggml_tensor *repeat_dummy =
      ggml_new_tensor_3d(ctx0, inpL->type, head_dim, N + n_past, n_head);

  struct ggml_tensor *layernorm_output;

  ggml_type wtype = GGML_TYPE_F32;
  // ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)
  // (model.hparams.ftype));
  const int sizeof_wtype = ggml_type_sizef(wtype);

  // const int i_gpu_start = n_layer - n_gpu_layers;
  const int i_gpu_start = lctx.model.i_gpu_start;
  const int i_gpu_last =
      lctx.model.i_gpu_last > 0 ? lctx.model.i_gpu_last : n_layer;
  (void)i_gpu_start;

  // offload functions set the tensor output backend to GPU
  // tensors are GPU-accelerated if any input or the output has been offloaded
  //
  // with the low VRAM option VRAM scratch is disabled in
  // llama_load_model_internal in that case ggml_cuda_assign_buffers has no
  // effect
  offload_func_t offload_func_nr = llama_nop;  // nr = non-repeating
  offload_func_t offload_func_kqv = llama_nop;

#ifdef GGML_USE_CUBLAS
  // todo: use either a flag in model/params or a backend test to determine if
  // norm/output are on GPU
  if (n_gpu_layers > n_layer) {
    offload_func_nr = ggml_cuda_assign_buffers;
  }
  if (n_gpu_layers > n_layer + 1) {
    offload_func_kqv = ggml_cuda_assign_buffers;
  }
#endif  // GGML_USE_CUBLAS

  for (int il = 0; il < n_layer; ++il) {
    offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (il >= i_gpu_start && il < i_gpu_last) {
      offload_func =
          ggml_cuda_assign_buffers;  // sets the output backend to GPU
    }
#endif  // GGML_USE_CUBLAS

    struct ggml_tensor *inpSA = inpL;

    lctx.use_buf(ctx0, 0);

    // self-attention
    {
      layernorm_output = ggml_norm(ctx0, inpL);

      ggml_tensor *il_a =
          ggml_mul(ctx0, layernorm_output, model.layers[il].input_layernorm);
      offload_func(il_a);  // (todo: uses vram scratch)

      layernorm_output =
          ggml_add(ctx0, il_a,
                   ggml_repeat(ctx0, model.layers[il].input_layernorm_b,
                               layernorm_output));
      offload_func(layernorm_output);
      ggml_set_name(layernorm_output, "layernorm_output");

      if (model.type == FALCON_40B || n_falcon_type == 40) {
        cur = ggml_norm(ctx0, inpL);

        cur = ggml_add(
            ctx0,
            ggml_mul(ctx0,
                     ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                     cur),
            ggml_repeat(ctx0, model.layers[il].attention_norm_b, cur));
      } else {
        cur = layernorm_output;
      }

      // compute QKV

      cur = ggml_mul_mat(ctx0, model.layers[il].query_key_value, cur);
      // offload_func(cur);

      // Note that the strides for Kcur, Vcur are set up so that the
      // resulting views are misaligned with the tensor's storage
      // (by applying the K/V offset we shift the tensor's original
      // view to stick out behind the viewed QKV tensor's allocated
      // memory, so to say). This is ok because no actual accesses
      // happen to that out-of-range memory, but it can require some
      // trickery when trying to accurately dump these views for
      // debugging.

      struct ggml_tensor *Qcur =
          ggml_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof_wtype,
                       head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype, 0);
      ggml_set_name(Qcur, "Qcur");

      struct ggml_tensor *Kcur = ggml_view_3d(
          ctx0, cur, head_dim, n_head_kv, N, head_dim * sizeof_wtype,
          head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
          head_dim * n_head * sizeof_wtype);
      ggml_set_name(Kcur, "Kcur");

      struct ggml_tensor *Vcur = ggml_view_3d(
          ctx0, cur, head_dim, n_head_kv, N, head_dim * sizeof_wtype,
          head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
          head_dim * (n_head + n_head_kv) * sizeof_wtype);
      ggml_set_name(Vcur, "Vcur");

      // using mode = 2 for neox mode
      Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, head_dim, 2);
      Kcur = ggml_rope_inplace(ctx0, Kcur, n_past, head_dim, 2);

      // store key and value to memory
      {
        struct ggml_tensor *k =
            ggml_view_1d(ctx0, kv_self.k, N * n_head_kv * head_dim,
                         (ggml_element_size(kv_self.k) * n_head_kv * head_dim) *
                             (il * n_ctx + n_past));
        ggml_set_name(k, "k");
        struct ggml_tensor *v =
            ggml_view_1d(ctx0, kv_self.v, N * n_head_kv * head_dim,
                         (ggml_element_size(kv_self.v) * n_head_kv * head_dim) *
                             (il * n_ctx + n_past));
        ggml_set_name(v, "v");

        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
      }

      struct ggml_tensor *K = ggml_permute(
          ctx0,
          ggml_reshape_3d(
              ctx0,
              ggml_view_1d(ctx0, kv_self.k, (n_past + N) * n_head_kv * head_dim,
                           il * n_ctx * ggml_element_size(kv_self.k) *
                               n_head_kv * head_dim),
              head_dim, n_head_kv, n_past + N),
          0, 2, 1, 3);

      // K * Q

      K = ggml_cont(ctx0, ggml_repeat2(ctx0, K, repeat_dummy));
      ggml_set_name(K, "K");

      struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
      ggml_set_name(Q, "Q");
      struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
      ggml_set_name(KQ, "KQ");

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ggml_tensor *KQ_scaled = ggml_scale_inplace(
          ctx0, KQ, ggml_new_f32(ctx0, 1.0f / sqrt(float(head_dim))));
      ggml_set_name(KQ_scaled, "KQ_scaled");

      // KQ_masked = mask_past(KQ_scaled)
      struct ggml_tensor *KQ_masked =
          ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
      ggml_set_name(KQ_masked, "KQ_masked");

      // KQ = soft_max(KQ_masked)
      struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
      ggml_set_name(KQ_soft_max, "KQ_soft_max");

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0,
      // 3).contiguous()
      struct ggml_tensor *V = ggml_permute(
          ctx0,
          ggml_reshape_3d(
              ctx0,
              ggml_view_1d(ctx0, kv_self.v, (n_past + N) * n_head_kv * head_dim,
                           il * n_ctx * ggml_element_size(model.kv_self.v) *
                               n_head_kv * head_dim),
              head_dim, n_head_kv, n_past + N),
          0, 2, 1, 3);

      V = ggml_cont(ctx0,
                    ggml_transpose(ctx0, ggml_repeat2(ctx0, V, repeat_dummy)));
      ggml_set_name(V, "V");

      // KQV = transpose(V) * KQ_soft_max
      struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
      ggml_set_name(KQV, "KQV");

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
      ggml_set_name(KQV_merged, "KQV_merged");

      // cur = KQV_merged.contiguous().view(n_embd, N)
      cur = ggml_cpy(ctx0, KQV_merged,
                     ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

      // projection
      {
        cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
        // offload_func(cur);
        ggml_set_name(cur, "result_wo");
      }
    }  // end of attention

    lctx.use_buf(ctx0, 1);
    // ggml_cuda_set_scratch(1);

    struct ggml_tensor *inpFF = layernorm_output;
    ggml_set_name(inpFF, "inpFF");
    struct ggml_tensor *attn_out =
        ggml_cpy(ctx0, cur, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
    // offload_func(attn_out);
    ggml_set_name(attn_out, "attn_out");
    {
      cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up, inpFF);
      // offload_func(cur);
      ggml_set_name(cur, "inpFF*ff_up");
      cur = ggml_gelu(ctx0, cur);
      // offload_func(cur);
      cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);
      // offload_func(cur);
      ggml_set_name(cur, "gelu_cur*ff_down");
    }

    cur = ggml_add(ctx0, cur, attn_out);
    cur = ggml_add(ctx0, cur, inpL);
    ggml_set_name(cur, "inpFF_+_result_attn_out");
    // input for next layer
    inpL = cur;
  }  // end of layer loop
  lctx.use_buf(ctx0, 0);
  // ggml_cuda_set_scratch(0);

  // used at the end to optionally extract the embeddings
  struct ggml_tensor *embeddings = NULL;

  offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
  if (n_gpu_layers > 0 && n_layer >= i_gpu_start && n_layer <= i_gpu_last) {
    offload_func = ggml_cuda_assign_buffers;  // sets the output backend to GPU
  }
#endif  // GGML_USE_CUBLAS

  // norm
  {
    cur = ggml_norm(ctx0, cur);
    // offload_func(cur);
    ggml_set_name(cur, "norm_cur");

    // inpL = ln_f_g*inpL + ln_f_b
    cur = ggml_add(
        ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.output_norm, cur), cur),
        ggml_repeat(ctx0, model.output_norm_b, cur));
    // offload_func(cur);
    ggml_set_name(cur, "result_norm");

    embeddings = cur;
  }

  // language modelling head
  cur = ggml_mul_mat(ctx0, model.lm_head, cur);
  // offload_func(cur);
  ggml_set_name(cur, "result_lm_head");

  //  cur = ggml_mul_mat(ctx0, model.output, cur);
  // ggml_set_name(cur, "result_output");

  lctx.use_buf(ctx0, -1);

  // logits -> probs
  // cur = ggml_soft_max_inplace(ctx0, cur);

  // run the computation
  ggml_build_forward_expand(&gf, cur);
#if 0
    // use to confirm vram_overhead is correct
    size_t vram_total=0;
    size_t vram_free=0;
#if defined(GGML_USE_CUBLAS)
    cudaMemGetInfo(&vram_free, &vram_total); // this should go in ggml-cuda.cu but I don't want to make Johannes life harder by modifying that yet
    fprintf(stderr, "\n%s: VRAM free: %7.2f MB  of %7.2f MB (in use: %7.2f MB)\n", __func__, vram_free/MB*1.0, vram_total/MB*1.0, (vram_total-vram_free)/MB*1.0);
#endif
#endif

#ifdef GGML_USE_METAL
  if (lctx.ctx_metal && N == 1) {
    ggml_metal_graph_compute(lctx.ctx_metal, &gf);
    ggml_metal_get_tensor(lctx.ctx_metal, cur);
  } else {
    // IMPORTANT:
    // Since we don't have efficient Matrix x Matrix Metal multiplication yet,
    // we fallback to vanilla ggml_graph_compute(). It uses Apple's Accelerate
    // CBLAS API which takes advantage of the ANE or the AMX coprocessor.
    //
    // When we implement Matrix x Matrix Metal multiplication, we can avoid this
    // branch. But for now, we have focused only on Matrix x Vector Metal
    // multiplication.
    //
    // TODO: avoid these syncs via shared memory (ref #1696)
    //
    if (lctx.ctx_metal) {
      // We need to sync the GPU KV cache with the CPU KV cache
      ggml_metal_get_tensor(lctx.ctx_metal, kv_self.k);
      ggml_metal_get_tensor(lctx.ctx_metal, kv_self.v);
    }

    ggml_graph_compute(ctx0, &gf);
  }
#else
  ggml_graph_compute(ctx0, &gf);
#endif

  if (cgraph_fname) {
    ggml_graph_export(&gf, cgraph_fname);
  }

  // plot the computation graph in dot format (for debugging purposes)
  // if (n_past%100 == 0) {
  //    ggml_graph_dump_dot(&gf, NULL, "llama.dot");
  //}

  // embd_w.resize(n_vocab*N);
  // memcpy(embd_w.data(), ggml_get_data(cur), sizeof(float)*n_vocab*N);

  // update kv token count
  lctx.model.kv_self.n = n_past + N;

  // extract logits
  {
    auto &logits_out = lctx.logits;

    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N);
      memcpy(logits_out.data(), (float *)ggml_get_data(cur),
             sizeof(float) * n_vocab * N);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(),
             (float *)ggml_get_data(cur) + (n_vocab * (N - 1)),
             sizeof(float) * n_vocab);
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto &embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(),
           (float *)ggml_get_data(embeddings) + (n_embd * (N - 1)),
           sizeof(float) * n_embd);
  }

  if (mem_per_token == 0) {
    mem_per_token = ggml_used_mem(ctx0) / N;
  }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

  ggml_free(ctx0);

  // measure the performance only for the single-token evals
  if (N == 1) {
    lctx.t_eval_us += ggml_time_us() - t_start_us;
    lctx.n_eval++;
  } else if (N > 1) {
    lctx.t_p_eval_us += ggml_time_us() - t_start_us;
    lctx.n_p_eval += N;
  }

  return true;
}

//
// tokenizer
//

static std::vector<llama_vocab::id> falcon_tokenize(const llama_vocab &vocab,
                                                    const std::string &text,
                                                    bool bos) {
  llama_tokenizer tokenizer(vocab);
  std::vector<llama_vocab::id> output;

  if (text.empty()) {
    return output;
  }

  if (bos) {
    output.push_back(falcon_token_bos());
  }

  tokenizer.tokenize(text, output);
  return output;
}

//
// sampling
//

void falcon_sample_softmax(struct falcon_context *ctx,
                           llama_token_data_array *candidates) {
  assert(candidates->size > 0);

  const int64_t t_start_sample_us = ggml_time_us();

  // Sort the logits in descending order
  if (!candidates->sorted) {
    std::sort(candidates->data, candidates->data + candidates->size,
              [](const llama_token_data &a, const llama_token_data &b) {
                return a.logit > b.logit;
              });
    candidates->sorted = true;
  }

  float max_l = candidates->data[0].logit;
  float cum_sum = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    float p = expf(candidates->data[i].logit - max_l);
    candidates->data[i].p = p;
    cum_sum += p;
  }
  for (size_t i = 0; i < candidates->size; ++i) {
    candidates->data[i].p /= cum_sum;
  }

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_top_k(struct falcon_context *ctx,
                         llama_token_data_array *candidates, int k,
                         size_t min_keep) {
  const int64_t t_start_sample_us = ggml_time_us();

  k = std::max(k, (int)min_keep);
  k = std::min(k, (int)candidates->size);

  // Sort scores in descending order
  if (!candidates->sorted) {
    auto comp = [](const llama_token_data &a, const llama_token_data &b) {
      return a.logit > b.logit;
    };
    if (k == (int)candidates->size) {
      std::sort(candidates->data, candidates->data + candidates->size, comp);
    } else {
      std::partial_sort(candidates->data, candidates->data + k,
                        candidates->data + candidates->size, comp);
    }
    candidates->sorted = true;
  }
  candidates->size = k;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_top_p(struct falcon_context *ctx,
                         llama_token_data_array *candidates, float p,
                         size_t min_keep) {
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  falcon_sample_softmax(ctx, candidates);

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;

  for (size_t i = 0; i < candidates->size; ++i) {
    cum_sum += candidates->data[i].p;

    // Check if the running sum is greater than p or if we have kept at least
    // min_keep tokens
    if (cum_sum > p && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the top-p tokens
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_tail_free(struct falcon_context *ctx,
                             llama_token_data_array *candidates, float z,
                             size_t min_keep) {
  if (z >= 1.0f || candidates->size <= 2) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  falcon_sample_softmax(nullptr, candidates);

  // Compute the first and second derivatives
  std::vector<float> first_derivatives(candidates->size - 1);
  std::vector<float> second_derivatives(candidates->size - 2);

  for (size_t i = 0; i < first_derivatives.size(); ++i) {
    first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
  }
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
  }

  // Calculate absolute value of second derivatives
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    second_derivatives[i] = abs(second_derivatives[i]);
  }

  // Normalize the second derivatives
  float second_derivatives_sum = std::accumulate(
      second_derivatives.begin(), second_derivatives.end(), 0.0f);
  for (float &value : second_derivatives) {
    value /= second_derivatives_sum;
  }

  float cum_sum = 0.0f;
  size_t last_idx = candidates->size;
  for (size_t i = 0; i < second_derivatives.size(); ++i) {
    cum_sum += second_derivatives[i];

    // Check if the running sum is greater than z or if we have kept at least
    // min_keep tokens
    if (cum_sum > z && i >= min_keep) {
      last_idx = i;
      break;
    }
  }

  // Resize the output vector to keep only the tokens above the tail location
  candidates->size = last_idx;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_typical(struct falcon_context *ctx,
                           llama_token_data_array *candidates, float p,
                           size_t min_keep) {
  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (p >= 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  // Compute the softmax of logits and calculate entropy
  falcon_sample_softmax(nullptr, candidates);

  float entropy = 0.0f;
  for (size_t i = 0; i < candidates->size; ++i) {
    entropy += -candidates->data[i].p * logf(candidates->data[i].p);
  }

  // Compute the absolute difference between negative log probability and
  // entropy for each candidate
  std::vector<float> shifted_scores;
  for (size_t i = 0; i < candidates->size; ++i) {
    float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
    shifted_scores.push_back(shifted_score);
  }

  // Sort tokens based on the shifted_scores and their corresponding indices
  std::vector<size_t> indices(candidates->size);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return shifted_scores[a] < shifted_scores[b];
  });

  // Compute the cumulative probabilities
  float cum_sum = 0.0f;
  size_t last_idx = indices.size();

  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    cum_sum += candidates->data[idx].p;

    // Check if the running sum is greater than typical or if we have kept at
    // least min_keep tokens
    if (cum_sum > p && i >= min_keep - 1) {
      last_idx = i + 1;
      break;
    }
  }

  // Resize the output vector to keep only the locally typical tokens
  std::vector<llama_token_data> new_candidates;
  for (size_t i = 0; i < last_idx; ++i) {
    size_t idx = indices[i];
    new_candidates.push_back(candidates->data[idx]);
  }

  // Replace the data in candidates with the new_candidates data
  std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
  candidates->size = new_candidates.size();

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_temperature(struct falcon_context *ctx,
                               llama_token_data_array *candidates_p,
                               float temp) {
  const int64_t t_start_sample_us = ggml_time_us();

  for (size_t i = 0; i < candidates_p->size; ++i) {
    candidates_p->data[i].logit /= temp;
  }

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_repetition_penalty(struct falcon_context *ctx,
                                      llama_token_data_array *candidates,
                                      const llama_token *last_tokens,
                                      size_t last_tokens_size, float penalty) {
  if (last_tokens_size == 0 || penalty == 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  for (size_t i = 0; i < candidates->size; ++i) {
    const auto *token_iter = std::find(
        last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
    if (token_iter == last_tokens + last_tokens_size) {
      continue;
    }

    // The academic publication that described this technique actually just only
    // divided, but that would cause tokens with negative logits to become more
    // likely, which is obviously wrong. This is common fix for this problem,
    // which is to multiply by the penalty instead of dividing.
    if (candidates->data[i].logit <= 0) {
      candidates->data[i].logit *= penalty;
    } else {
      candidates->data[i].logit /= penalty;
    }
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_frequency_and_presence_penalties(
    struct falcon_context *ctx, llama_token_data_array *candidates,
    const llama_token *last_tokens_p, size_t last_tokens_size,
    float alpha_frequency, float alpha_presence) {
  if (last_tokens_size == 0 ||
      (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  // Create a frequency map to count occurrences of each token in last_tokens
  std::unordered_map<llama_token, int> token_count;
  for (size_t i = 0; i < last_tokens_size; ++i) {
    token_count[last_tokens_p[i]]++;
  }

  // Apply frequency and presence penalties to the candidates
  for (size_t i = 0; i < candidates->size; ++i) {
    auto token_iter = token_count.find(candidates->data[i].id);
    if (token_iter == token_count.end()) {
      continue;
    }

    int count = token_iter->second;
    candidates->data[i].logit -=
        float(count) * alpha_frequency + float(count > 0) * alpha_presence;
  }

  candidates->sorted = false;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

llama_token falcon_sample_token_mirostat(struct falcon_context *ctx,
                                         llama_token_data_array *candidates,
                                         float tau, float eta, int m,
                                         float *mu) {
  assert(ctx);
  auto N = float(falcon_n_vocab(ctx));
  int64_t t_start_sample_us;
  t_start_sample_us = ggml_time_us();

  falcon_sample_softmax(nullptr, candidates);

  // Estimate s_hat using the most probable m tokens
  float s_hat = 0.0;
  float sum_ti_bi = 0.0;
  float sum_ti_sq = 0.0;
  for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
    float t_i = logf(float(i + 2) / float(i + 1));
    float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
    sum_ti_bi += t_i * b_i;
    sum_ti_sq += t_i * t_i;
  }
  s_hat = sum_ti_bi / sum_ti_sq;

  // Compute k from the estimated s_hat and target surprise value
  float epsilon_hat = s_hat - 1;
  float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)),
                 1 / s_hat);

  // Sample the next word X using top-k sampling
  falcon_sample_top_k(nullptr, candidates, int(k), 1);
  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
  llama_token X = falcon_sample_token(ctx, candidates);
  t_start_sample_us = ggml_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const llama_token_data &candidate) {
                     return candidate.id == X;
                   }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return X;
}

llama_token falcon_sample_token_mirostat_v2(struct falcon_context *ctx,
                                            llama_token_data_array *candidates,
                                            float tau, float eta, float *mu) {
  assert(ctx);
  int64_t t_start_sample_us;
  t_start_sample_us = ggml_time_us();

  falcon_sample_softmax(ctx, candidates);

  // Truncate the words with surprise values greater than mu
  candidates->size = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const llama_token_data &candidate) {
                     return -log2f(candidate.p) > *mu;
                   }));

  if (candidates->size == 0) {
    candidates->size = 1;
  }

  // Normalize the probabilities of the remaining words
  falcon_sample_softmax(ctx, candidates);

  // Sample the next word X from the remaining words
  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
  llama_token X = falcon_sample_token(ctx, candidates);
  t_start_sample_us = ggml_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const llama_token_data &candidate) {
                     return candidate.id == X;
                   }));
  float observed_surprise = -log2f(candidates->data[X_idx].p);
  float e = observed_surprise - tau;

  // Update mu using the learning rate and error
  *mu = *mu - eta * e;

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
  return X;
}

llama_token falcon_sample_token_greedy(struct falcon_context *ctx,
                                       llama_token_data_array *candidates) {
  const int64_t t_start_sample_us = ggml_time_us();

  // Find max element
  auto *max_iter = std::max_element(
      candidates->data, candidates->data + candidates->size,
      [](const llama_token_data &a, const llama_token_data &b) {
        return a.logit < b.logit;
      });

  llama_token result = max_iter->id;
  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return result;
}

llama_token falcon_sample_token(struct falcon_context *ctx,
                                llama_token_data_array *candidates) {
  assert(ctx);
  const int64_t t_start_sample_us = ggml_time_us();
  falcon_sample_softmax(nullptr, candidates);

  std::vector<float> probs;
  probs.reserve(candidates->size);
  for (size_t i = 0; i < candidates->size; ++i) {
    probs.push_back(candidates->data[i].p);
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  auto &rng = ctx->rng;
  int idx = dist(rng);

  llama_token result = candidates->data[idx].id;

  ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  ctx->n_sample++;
  return result;
}

//
// quantization
//

static void falcon_model_quantize_internal(
    const std::string &fname_inp, const std::string &fname_out,
    const falcon_model_quantize_params *params) {
  ggml_type quantized_type;
  llama_ftype ftype = params->ftype;
  int nthread = params->nthread;

  switch (params->ftype) {
    case LLAMA_FTYPE_MOSTLY_Q4_0:
      quantized_type = GGML_TYPE_Q4_0;
      break;
    case LLAMA_FTYPE_MOSTLY_Q4_1:
      quantized_type = GGML_TYPE_Q4_1;
      break;
    case LLAMA_FTYPE_MOSTLY_Q5_0:
      quantized_type = GGML_TYPE_Q5_0;
      break;
    case LLAMA_FTYPE_MOSTLY_Q5_1:
      quantized_type = GGML_TYPE_Q5_1;
      break;
    case LLAMA_FTYPE_MOSTLY_Q8_0:
      quantized_type = GGML_TYPE_Q8_0;
      break;
    case LLAMA_FTYPE_MOSTLY_F16:
      quantized_type = GGML_TYPE_F16;
      break;
    case LLAMA_FTYPE_ALL_F32:
      quantized_type = GGML_TYPE_F32;
      break;

#ifdef GGML_USE_K_QUANTS
    // K-quants
    case LLAMA_FTYPE_MOSTLY_Q2_K:
      quantized_type = GGML_TYPE_Q2_K;
      break;
    case LLAMA_FTYPE_MOSTLY_Q3_K_S:
    case LLAMA_FTYPE_MOSTLY_Q3_K_M:
    case LLAMA_FTYPE_MOSTLY_Q3_K_L:
      quantized_type = GGML_TYPE_Q3_K;
      break;
    case LLAMA_FTYPE_MOSTLY_Q4_K_S:
    case LLAMA_FTYPE_MOSTLY_Q4_K_M:
      quantized_type = GGML_TYPE_Q4_K;
      break;
    case LLAMA_FTYPE_MOSTLY_Q5_K_S:
    case LLAMA_FTYPE_MOSTLY_Q5_K_M:
      quantized_type = GGML_TYPE_Q5_K;
      break;
    case LLAMA_FTYPE_MOSTLY_Q6_K:
      quantized_type = GGML_TYPE_Q6_K;
      break;
#endif
    default:
      throw std::runtime_error(format("invalid output file type %d\n", ftype));
  }

  if (nthread <= 0) {
    nthread = std::thread::hardware_concurrency();
  }

  std::unique_ptr<falcon_model_loader> model_loader(
      new falcon_model_loader(fname_inp, /*use_mmap*/ false,
                              /*vocab_only*/ false));
  falcon_file_saver file_saver(
      fname_out.c_str(), model_loader->file_loaders.at(0).get(), params->ftype);

#ifdef GGML_USE_K_QUANTS
  int n_attention_wv = 0;
  int n_feed_forward_w2 = 0;
  for (auto &tensor : model_loader->tensors_map.tensors) {
    if (tensor.name.find("attention.wv.weight") != std::string::npos) {
      ++n_attention_wv;
    } else if (tensor.name.find("feed_forward.w2.weight") !=
               std::string::npos) {
      ++n_feed_forward_w2;
    }
  }

  int i_attention_wv = 0;
  int i_feed_forward_w2 = 0;
#endif

  size_t total_size_org = 0;
  size_t total_size_new = 0;
  std::vector<int64_t> hist_all(1 << 4, 0);

  std::vector<std::thread> workers;
  std::mutex mutex;

  size_t idx = 0;
  for (llama_load_tensor &tensor : model_loader->tensors_map.tensors) {
    llama_buffer read_data;
    read_data.resize(tensor.size);
    tensor.data = read_data.addr;
    model_loader->load_data_for(tensor);

    printf("[%4zu/%4zu] %36s - %16s, type = %6s, ", ++idx,
           model_loader->tensors_map.tensors.size(), tensor.name.c_str(),
           llama_format_tensor_shape(tensor.ne).c_str(),
           ggml_type_name(tensor.type));

    // This used to be a regex, but <regex> has an extreme cost to compile
    // times.
    bool quantize = tensor.name.rfind("weight") ==
                    tensor.name.size() - 6;  // ends with 'weight'?

    // quantize only 2D tensors
    quantize &= (tensor.ne.size() == 2);
    quantize &=
        params->quantize_output_tensor || tensor.name != "output.weight";
    quantize &= quantized_type != tensor.type;

    enum ggml_type new_type;
    void *new_data;
    size_t new_size;
    llama_buffer work;

    if (!quantize) {
      new_type = tensor.type;
      new_data = tensor.data;
      new_size = tensor.size;
      printf("(Not quantizing) size = %8.3f MB\n",
             tensor.size / 1024.0 / 1024.0);
    } else {
      new_type = quantized_type;
#ifdef GGML_USE_K_QUANTS
      if (quantized_type == GGML_TYPE_Q2_K ||
          quantized_type == GGML_TYPE_Q3_K ||
          quantized_type == GGML_TYPE_Q4_K ||
          quantized_type == GGML_TYPE_Q5_K ||
          quantized_type == GGML_TYPE_Q6_K) {
        int nx = tensor.ne.at(0);
        int ny = tensor.ne.at(0);
        if (nx % QK_K != 0 || ny % QK_K != 0) {
          fprintf(stderr,
                  "\n\n========================= Tensor sizes %d x %d are not "
                  "divisible by %d\n",
                  nx, ny, QK_K);
          fprintf(stderr,
                  "This is required to be able to use k-quants for now!\n");
          fprintf(stderr,
                  "============================================================"
                  "============================\n\n");
          throw std::runtime_error("Unsupported tensor size encountered\n");
        }
      }
      // if (tensor.name == ".mlp.dense_") {
      //    new_type = GGML_TYPE_Q6_K;
      // }
      // TODO falcon

#endif

      float *f32_data;
      size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
      llama_buffer f32_conv_buf;

      if (tensor.type == GGML_TYPE_F32) {
        f32_data = (float *)tensor.data;
      } else if (ggml_is_quantized(tensor.type) && !params->allow_requantize) {
        throw std::runtime_error(format("requantizing from type %s is disabled",
                                        ggml_type_name(tensor.type)));
      } else {
        llama_convert_tensor_internal(tensor, f32_conv_buf, nelements, nthread);
        f32_data = (float *)f32_conv_buf.addr;
      }

      printf("quantizing .. ");
      fflush(stdout);

      work.resize(nelements * 4);  // upper bound on size
      new_data = work.addr;
      std::vector<int64_t> hist_cur(1 << 4, 0);

      int chunk_size = 32 * 512;
      const int nchunk = (nelements + chunk_size - 1) / chunk_size;
      const int nthread_use =
          nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
      if (nthread_use < 2) {
        new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0,
                                       nelements, hist_cur.data());
      } else {
        size_t counter = 0;
        new_size = 0;
        auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type,
                        f32_data, new_data, nelements, chunk_size]() {
          std::vector<int64_t> local_hist;
          size_t local_size = 0;
          while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            size_t first = counter;
            counter += chunk_size;
            if (first >= nelements) {
              if (!local_hist.empty()) {
                for (int j = 0; j < int(local_hist.size()); ++j) {
                  hist_cur[j] += local_hist[j];
                }
                new_size += local_size;
              }
              break;
            }
            lock.unlock();
            size_t last = std::min(nelements, first + chunk_size);
            if (local_hist.empty()) {
              local_hist.resize(hist_cur.size(), 0);
            }
            local_size +=
                ggml_quantize_chunk(new_type, f32_data, new_data, first,
                                    last - first, local_hist.data());
          }
        };
        if ((int)workers.size() < nthread_use - 1) {
          workers.resize(nthread_use - 1);
        }
        for (int it = 0; it < nthread_use - 1; ++it) {
          workers[it] = std::thread(compute);
        }
        compute();
        for (int it = 0; it < nthread_use - 1; ++it) {
          workers[it].join();
        }
      }

      printf("size = %8.2f MB -> %8.2f MB | hist: ",
             tensor.size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
      int64_t tot_count = 0;
      for (size_t i = 0; i < hist_cur.size(); i++) {
        hist_all[i] += hist_cur[i];
        tot_count += hist_cur[i];
      }

      if (tot_count > 0) {
        for (size_t i = 0; i < hist_cur.size(); i++) {
          printf("%5.3f ", hist_cur[i] / float(nelements));
        }
      }
      printf("\n");
    }
    total_size_org += tensor.size;
    total_size_new += new_size;
    file_saver.write_tensor(tensor, new_type, new_data, new_size);
  }

  printf("%s: model size  = %8.2f MB\n", __func__,
         total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB\n", __func__,
         total_size_new / 1024.0 / 1024.0);

  {
    int64_t sum_all = 0;
    for (size_t i = 0; i < hist_all.size(); i++) {
      sum_all += hist_all[i];
    }

    if (sum_all > 0) {
      printf("%s: hist: ", __func__);
      for (size_t i = 0; i < hist_all.size(); i++) {
        printf("%5.3f ", hist_all[i] / float(sum_all));
      }
      printf("\n");
    }
  }
}

//
// interface implementation
//

struct falcon_context *falcon_init_from_file(
    const char *path_model, struct falcon_context_params params) {
  ggml_time_init();

  falcon_context *ctx = new falcon_context;

  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  unsigned cur_percentage = 0;
  if (params.progress_callback == NULL) {
    params.progress_callback_user_data = &cur_percentage;
    params.progress_callback = [](float progress, void *ctx) {
      unsigned *cur_percentage_p = (unsigned *)ctx;
      unsigned percentage = (unsigned)(100 * progress);
      while (percentage > *cur_percentage_p) {
        *cur_percentage_p = percentage;
      }
    };
  }

  ctx->rng = std::mt19937(params.seed);
  ctx->logits_all = params.logits_all;

  ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

  if (!falcon_model_load(
          path_model, *ctx, params.n_ctx, params.n_batch, params.n_gpu_layers,
          params.main_gpu, params.tensor_split, memory_type, params.use_mmap,
          params.use_mlock, params.vocab_only, params.progress_callback,
          params.progress_callback_user_data)) {
    fprintf(stderr, "%s: failed to load model\n", __func__);
    falcon_free(ctx);
    return nullptr;
  }
  // model_load_internal() may change this if VRAM runs out
  params.n_gpu_layers = ctx->model.n_gpu_layers;
  params.i_gpu_start =
      ctx->model.i_gpu_start;  // first layer that's GPU accelerated
  params.i_gpu_last =
      ctx->model.i_gpu_last;  // last layer that's GPU accelerated

  // reserve memory for context buffers
  if (!params.vocab_only) {
    if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type,
                       ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
      fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n",
              __func__);
      falcon_free(ctx);
      return nullptr;
    }

    const auto &hparams = ctx->model.hparams;

    // resized during inference
    if (params.logits_all) {
      ctx->logits.reserve(hparams.n_ctx * hparams.n_vocab);
    } else {
      ctx->logits.reserve(hparams.n_vocab);
    }

    if (params.embedding) {
      ctx->embedding.resize(hparams.n_embd);
    }

    ctx->buf_compute.resize(FALCON_MEM_REQ_EVAL().at(ctx->model.type));

    ctx->buf_scratch[0].resize(FALCON_MEM_REQ_SCRATCH0().at(ctx->model.type));
    ctx->buf_scratch[1].resize(FALCON_MEM_REQ_SCRATCH1().at(ctx->model.type));
  }

#ifdef GGML_USE_METAL
  if (params.n_gpu_layers > 0) {
    // this allocates all Metal resources and memory buffers
    ctx->ctx_metal = ggml_metal_init();

    void *data_ptr = NULL;
    size_t data_size = 0;
    if (params.use_mmap) {
      data_ptr = ctx->model.mapping->addr;
      data_size = ctx->model.mapping->size;
    } else {
      data_ptr = ggml_get_mem_buffer(ctx->model.ctx);
      data_size = ggml_get_mem_size(ctx->model.ctx);
    }

#define LLAMA_METAL_CHECK_BUF(result)                        \
  if (!(result)) {                                           \
    fprintf(stderr, "%s: failed to add buffer\n", __func__); \
    falcon_free(ctx);                                        \
    return NULL;                                             \
  }

    LLAMA_METAL_CHECK_BUF(
        ggml_metal_add_buffer(ctx->ctx_metal, "data", data_ptr, data_size));
    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(
        ctx->ctx_metal, "eval", ctx->buf_compute.addr, ctx->buf_compute.size));

    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "kv",
                                                ctx->model.kv_self.buf.addr,
                                                ctx->model.kv_self.buf.size));
    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr0",
                                                ctx->buf_scratch[0].addr,
                                                ctx->buf_scratch[0].size));
    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr1",
                                                ctx->buf_scratch[1].addr,
                                                ctx->buf_scratch[1].size));
#undef LLAMA_METAL_CHECK_BUF
  }
#endif

  return ctx;
}

void falcon_free(struct falcon_context *ctx) { delete ctx; }

int falcon_model_quantize(const char *fname_inp, const char *fname_out,
                          const falcon_model_quantize_params *params) {
  try {
    falcon_model_quantize_internal(fname_inp, fname_out, params);
    return 0;
  } catch (const std::exception &err) {
    fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.what());
    return 1;
  }
}

int falcon_apply_lora_from_file_internal(struct falcon_context *ctx,
                                         const char *path_lora,
                                         const char *path_base_model,
                                         int n_threads) {
  fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n",
          __func__, path_lora);

  auto &model = ctx->model;

  const int64_t t_start_lora_us = ggml_time_us();

  auto fin = std::ifstream(path_lora, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
    return 1;
  }

  // verify magic and version
  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != LLAMA_FILE_MAGIC_GGLA) {
      fprintf(stderr, "%s: bad file magic\n", __func__);
      return 1;
    }
    uint32_t format_version;
    fin.read((char *)&format_version, sizeof(format_version));

    if (format_version != 1) {
      fprintf(stderr, "%s: unsupported file version\n", __func__);
      return 1;
    }
  }

  int32_t lora_r;
  int32_t lora_alpha;
  fin.read((char *)&lora_r, sizeof(lora_r));
  fin.read((char *)&lora_alpha, sizeof(lora_alpha));
  float scaling = (float)lora_alpha / (float)lora_r;

  fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r,
          lora_alpha, scaling);

  // create a temporary ggml context to store the lora tensors
  // todo: calculate size from biggest possible tensor
  std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
  struct ggml_init_params params;
  params.mem_size = lora_buf.size();
  params.mem_buffer = lora_buf.data();
  params.no_alloc = false;

  ggml_context *lora_ctx = ggml_init(params);
  std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

  // create a name -> tensor map of the model to accelerate lookups
  std::unordered_map<std::string, struct ggml_tensor *> model_tensors;
  for (auto &kv : model.tensors_by_name) {
    model_tensors.insert(kv);
  }

  // load base model
  std::unique_ptr<falcon_model_loader> model_loader;
  ggml_context *base_ctx = NULL;
  llama_buffer base_buf;
  if (path_base_model) {
    fprintf(stderr, "%s: loading base model from '%s'\n", __func__,
            path_base_model);
    model_loader.reset(new falcon_model_loader(
        path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

    size_t ctx_size;
    size_t mmapped_size;
    model_loader->calc_sizes(&ctx_size, &mmapped_size);
    base_buf.resize(ctx_size);

    ggml_init_params base_params;
    base_params.mem_size = base_buf.size;
    base_params.mem_buffer = base_buf.addr;
    base_params.no_alloc = model_loader->use_mmap;

    base_ctx = ggml_init(base_params);

    model_loader->ggml_ctx = base_ctx;

    // maybe this should in falcon_model_loader
    if (model_loader->use_mmap) {
      model_loader->mapping.reset(new llama_mmap(
          &model_loader->file_loaders.at(0)->file, /* prefetch */ 0));
    }
  }

  // read tensors and apply
  bool warned = false;
  int n_tensors = 0;
  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ftype;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));
    if (fin.eof()) {
      break;
    }

    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
    }

    std::string name;
    {
      char buf[1024];
      fin.read(buf, length);
      name = std::string(buf, length);
    }

    // check for lora suffix and get the type of tensor
    const std::string lora_suffix = ".lora";
    size_t pos = name.rfind(lora_suffix);
    if (pos == std::string::npos) {
      fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__,
              name.c_str());
      return 1;
    }

    std::string lora_type = name.substr(pos + lora_suffix.length());
    std::string base_name = name;
    base_name.erase(pos);
    // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__,
    // name.c_str(),base_name.c_str(), lora_type.c_str());

    if (model_tensors.find(base_name) == model_tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__,
              name.data());
      return 1;
    }

    // create ggml tensor
    ggml_type wtype;
    switch (ftype) {
      case 0:
        wtype = GGML_TYPE_F32;
        break;
      case 1:
        wtype = GGML_TYPE_F16;
        break;
      default: {
        fprintf(stderr, "%s: invalid tensor data type '%d'\n", __func__, ftype);
        return false;
      }
    }
    ggml_tensor *lora_tensor;
    if (n_dims == 2) {
      lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
    } else {
      fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__,
              n_dims);
      return 1;
    }

    // load tensor data
    size_t offset = fin.tellg();
    size_t tensor_data_size = ggml_nbytes(lora_tensor);
    offset = (offset + 31) & -32;
    fin.seekg(offset);
    fin.read((char *)lora_tensor->data, tensor_data_size);

    lora_tensors[name] = lora_tensor;

    // check if we have both A and B tensors and apply
    if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
        lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {
      ggml_tensor *dest_t = model_tensors[base_name];
      ggml_tensor *base_t;
      if (model_loader) {
        // load from base model
        if (model_loader->tensors_map.name_to_idx.find(base_name) ==
            model_loader->tensors_map.name_to_idx.end()) {
          fprintf(stderr, "%s: error: tensor '%s' not found in base model\n",
                  __func__, base_name.c_str());
          return 1;
        }
        size_t idx = model_loader->tensors_map.name_to_idx[base_name];
        llama_load_tensor &lt = model_loader->tensors_map.tensors[idx];
        base_t = model_loader->get_tensor(
            base_name, {(uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1]},
            GGML_BACKEND_CPU);
        lt.data = (uint8_t *)lt.ggml_tensor->data;
        model_loader->load_data_for(lt);
        lt.ggml_tensor->data = lt.data;
      } else {
        base_t = dest_t;
      }

      if (ggml_is_quantized(base_t->type)) {
        if (!warned) {
          fprintf(stderr,
                  "%s: warning: using a lora adapter with a quantized model "
                  "may result in poor quality, "
                  "use a f16 or f32 base model with --lora-base\n",
                  __func__);
          warned = true;
        }
      }

      ggml_tensor *loraA = lora_tensors[base_name + ".loraA"];
      ggml_tensor *loraB = lora_tensors[base_name + ".loraB"];

      if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
        fprintf(stderr,
                "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64
                ");"
                " are you sure that this adapter is for this model?\n",
                __func__, base_t->ne[0], loraA->ne[1]);
        return 1;
      }

      // w = w + BA*s
      ggml_tensor *BA = ggml_mul_mat(lora_ctx, loraA, loraB);

      if (scaling != 1.0f) {
        ggml_tensor *scale_tensor = ggml_new_f32(lora_ctx, scaling);
        BA = ggml_scale_inplace(lora_ctx, BA, scale_tensor);
      }

      ggml_tensor *r;
      if (base_t == dest_t) {
        r = ggml_add_inplace(lora_ctx, dest_t, BA);
      } else {
        r = ggml_add(lora_ctx, base_t, BA);
        r = ggml_cpy(lora_ctx, r, dest_t);
      }

      struct ggml_cgraph gf = ggml_build_forward(r);
      gf.n_threads = n_threads;
      ggml_graph_compute(lora_ctx, &gf);

      // we won't need these tensors again, reset the context to save memory
      ggml_free(lora_ctx);
      lora_ctx = ggml_init(params);
      lora_tensors.clear();

      n_tensors++;
      if (n_tensors % 4 == 0) {
        fprintf(stderr, ".");
      }
    }
  }

  // TODO: this should be in a destructor, it will leak on failure
  ggml_free(lora_ctx);
  if (base_ctx) {
    ggml_free(base_ctx);
  }

  const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
  fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

  return 0;
}

int falcon_apply_lora_from_file(struct falcon_context *ctx,
                                const char *path_lora,
                                const char *path_base_model, int n_threads) {
  try {
    return falcon_apply_lora_from_file_internal(ctx, path_lora, path_base_model,
                                                n_threads);
  } catch (const std::exception &err) {
    fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__,
            err.what());
    return 1;
  }
}

int falcon_get_kv_cache_token_count(const struct falcon_context *ctx) {
  return ctx->model.kv_self.n;
}

#define LLAMA_MAX_RNG_STATE (64 * 1024)

void falcon_set_rng_seed(struct falcon_context *ctx, int seed) {
  if (seed < 0) {
    seed = time(NULL);
  }
  ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t falcon_get_state_size(const struct falcon_context *ctx) {
  // we don't know size of rng until we actually serialize it. so reserve more
  // than enough memory for its serialized state. for reference,
  // std::mt19937(1337) serializes to 6701 bytes.
  const size_t s_rng_size = sizeof(size_t);
  const size_t s_rng = LLAMA_MAX_RNG_STATE;
  const size_t s_logits_capacity = sizeof(size_t);
  const size_t s_logits_size = sizeof(size_t);
  const size_t s_logits = ctx->logits.capacity() * sizeof(float);
  const size_t s_embedding_size = sizeof(size_t);
  const size_t s_embedding = ctx->embedding.size() * sizeof(float);
  const size_t s_kv_size = sizeof(size_t);
  const size_t s_kv_ntok = sizeof(int);
  const size_t s_kv = ctx->model.kv_self.buf.size;

  const size_t s_total =
      (+s_rng_size + s_rng + s_logits_capacity + s_logits_size + s_logits +
       s_embedding_size + s_embedding + s_kv_size + s_kv_ntok + s_kv);

  return s_total;
}

// Copies the state to the specified destination address
size_t falcon_copy_state_data(struct falcon_context *ctx, uint8_t *dst) {
  uint8_t *out = dst;

  // copy rng
  {
    std::stringstream rng_ss;
    rng_ss << ctx->rng;

    const size_t rng_size = rng_ss.str().size();
    char rng_buf[LLAMA_MAX_RNG_STATE];

    memset(&rng_buf[0], 0, LLAMA_MAX_RNG_STATE);
    memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

    memcpy(out, &rng_size, sizeof(rng_size));
    out += sizeof(rng_size);
    memcpy(out, &rng_buf[0], LLAMA_MAX_RNG_STATE);
    out += LLAMA_MAX_RNG_STATE;
  }

  // copy logits
  {
    const size_t logits_cap = ctx->logits.capacity();
    const size_t logits_size = ctx->logits.size();

    memcpy(out, &logits_cap, sizeof(logits_cap));
    out += sizeof(logits_cap);
    memcpy(out, &logits_size, sizeof(logits_size));
    out += sizeof(logits_size);

    if (logits_size) {
      memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
    }

    out += logits_cap * sizeof(float);
  }

  // copy embeddings
  {
    const size_t embedding_size = ctx->embedding.size();

    memcpy(out, &embedding_size, sizeof(embedding_size));
    out += sizeof(embedding_size);

    if (embedding_size) {
      memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
      out += embedding_size * sizeof(float);
    }
  }

  // copy kv cache
  {
    const auto &kv_self = ctx->model.kv_self;
    const auto &hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    const size_t kv_size = kv_self.buf.size;
    const int kv_ntok = falcon_get_kv_cache_token_count(ctx);

    memcpy(out, &kv_size, sizeof(kv_size));
    out += sizeof(kv_size);
    memcpy(out, &kv_ntok, sizeof(kv_ntok));
    out += sizeof(kv_ntok);

    if (kv_size) {
      const size_t elt_size = ggml_element_size(kv_self.k);

      char buffer[4096];

      ggml_context *cpy_ctx =
          ggml_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ggml_cgraph gf{};
      gf.n_threads = 1;

      ggml_tensor *kout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd,
                                               kv_ntok, n_layer);
      kout3d->data = out;
      out += ggml_nbytes(kout3d);

      ggml_tensor *vout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type,
                                               kv_ntok, n_embd, n_layer);
      vout3d->data = out;
      out += ggml_nbytes(vout3d);

      ggml_tensor *k3d =
          ggml_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer,
                       elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ggml_tensor *v3d =
          ggml_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer,
                       elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, k3d, kout3d));
      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, v3d, vout3d));
      ggml_graph_compute(cpy_ctx, &gf);

      ggml_free(cpy_ctx);
    }
  }

  const size_t written = out - dst;
  const size_t max_size = falcon_get_state_size(ctx);

  LLAMA_ASSERT(written <= max_size);

  return written;
}

// Sets the state reading from the specified source address
size_t falcon_set_state_data(struct falcon_context *ctx, uint8_t *src) {
  uint8_t *inp = src;

  // set rng
  {
    size_t rng_size;
    char rng_buf[LLAMA_MAX_RNG_STATE];

    memcpy(&rng_size, inp, sizeof(rng_size));
    inp += sizeof(rng_size);
    memcpy(&rng_buf[0], inp, LLAMA_MAX_RNG_STATE);
    inp += LLAMA_MAX_RNG_STATE;

    std::stringstream rng_ss;
    rng_ss.str(std::string(&rng_buf[0], rng_size));
    rng_ss >> ctx->rng;

    LLAMA_ASSERT(rng_ss.fail() == false);
  }

  // set logits
  {
    size_t logits_cap;
    size_t logits_size;

    memcpy(&logits_cap, inp, sizeof(logits_cap));
    inp += sizeof(logits_cap);
    memcpy(&logits_size, inp, sizeof(logits_size));
    inp += sizeof(logits_size);

    LLAMA_ASSERT(ctx->logits.capacity() == logits_cap);

    if (logits_size) {
      ctx->logits.resize(logits_size);
      memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
    }

    inp += logits_cap * sizeof(float);
  }

  // set embeddings
  {
    size_t embedding_size;

    memcpy(&embedding_size, inp, sizeof(embedding_size));
    inp += sizeof(embedding_size);

    LLAMA_ASSERT(ctx->embedding.capacity() == embedding_size);

    if (embedding_size) {
      memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
      inp += embedding_size * sizeof(float);
    }
  }

  // set kv cache
  {
    const auto &kv_self = ctx->model.kv_self;
    const auto &hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;

    size_t kv_size;
    int kv_ntok;

    memcpy(&kv_size, inp, sizeof(kv_size));
    inp += sizeof(kv_size);
    memcpy(&kv_ntok, inp, sizeof(kv_ntok));
    inp += sizeof(kv_ntok);

    if (kv_size) {
      LLAMA_ASSERT(kv_self.buf.size == kv_size);

      const size_t elt_size = ggml_element_size(kv_self.k);

      char buffer[4096];

      ggml_context *cpy_ctx =
          ggml_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ggml_cgraph gf{};
      gf.n_threads = 1;

      ggml_tensor *kin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd,
                                              kv_ntok, n_layer);
      kin3d->data = (void *)inp;
      inp += ggml_nbytes(kin3d);

      ggml_tensor *vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok,
                                              n_embd, n_layer);
      vin3d->data = (void *)inp;
      inp += ggml_nbytes(vin3d);

      ggml_tensor *k3d =
          ggml_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer,
                       elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ggml_tensor *v3d =
          ggml_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer,
                       elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, kin3d, k3d));
      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, vin3d, v3d));
      ggml_graph_compute(cpy_ctx, &gf);

      ggml_free(cpy_ctx);
    }

    ctx->model.kv_self.n = kv_ntok;
  }

  const size_t nread = inp - src;
  const size_t max_size = falcon_get_state_size(ctx);

  LLAMA_ASSERT(nread <= max_size);

  return nread;
}

bool falcon_load_session_file(struct falcon_context *ctx,
                              const char *path_session, llama_token *tokens_out,
                              size_t n_token_capacity,
                              size_t *n_token_count_out) {
  llama_file file(path_session, "rb");

  // sanity checks
  {
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
      fprintf(stderr,
              "%s : unknown (magic, version) for session file: %08x, %08x\n",
              __func__, magic, version);
      return false;
    }

    falcon_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(falcon_hparams));

    if (session_hparams != ctx->model.hparams) {
      fprintf(stderr, "%s : model hparams didn't match from session file!\n",
              __func__);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t n_token_count = file.read_u32();

    if (n_token_count > n_token_capacity) {
      fprintf(stderr,
              "%s : token count in session file exceeded capacity! %u > %zu\n",
              __func__, n_token_count, n_token_capacity);
      return false;
    }

    file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
    *n_token_count_out = n_token_count;
  }

  // restore the context state
  {
    const size_t n_state_size_cur = file.size - file.tell();
    const size_t n_state_size_max = falcon_get_state_size(ctx);

    if (n_state_size_cur > n_state_size_max) {
      fprintf(
          stderr,
          "%s : the state size in session file is too big! max %zu, got %zu\n",
          __func__, n_state_size_max, n_state_size_cur);
      return false;
    }

    std::vector<uint8_t> state_data(n_state_size_max);
    file.read_raw(state_data.data(), n_state_size_cur);

    falcon_set_state_data(ctx, state_data.data());
  }

  return true;
}

bool falcon_save_session_file(struct falcon_context *ctx,
                              const char *path_session,
                              const llama_token *tokens, size_t n_token_count) {
  llama_file file(path_session, "wb");

  file.write_u32(LLAMA_SESSION_MAGIC);
  file.write_u32(LLAMA_SESSION_VERSION);

  file.write_raw(&ctx->model.hparams, sizeof(falcon_hparams));

  // save the prompt
  file.write_u32((uint32_t)n_token_count);
  file.write_raw(tokens, sizeof(llama_token) * n_token_count);

  // save the context state
  {
    const size_t n_state_size_max = falcon_get_state_size(ctx);

    std::vector<uint8_t> state_data(n_state_size_max);
    const size_t n_state_size_cur =
        falcon_copy_state_data(ctx, state_data.data());

    file.write_raw(state_data.data(), n_state_size_cur);
  }

  return true;
}

int falcon_eval(struct falcon_context *ctx, const llama_token *tokens,
                int n_tokens, int n_past, int n_threads, int debug_timings) {
  if (!falcon_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads, nullptr,
                            debug_timings)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  // get a more accurate load time, upon first eval
  // TODO: fix this
  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}

int falcon_eval_export(struct falcon_context *ctx, const char *fname) {
  const int n_batch = 1;
  const int n_ctx = 512 - n_batch;

  const std::vector<llama_token> tmp(n_batch, falcon_token_bos());

  if (!falcon_eval_internal(*ctx, tmp.data(), tmp.size(), n_ctx, 1, fname, 0)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  return 0;
}

int falcon_tokenize(struct falcon_context *ctx, const char *text,
                    llama_token *tokens, int n_max_tokens, bool add_bos) {
  auto res = falcon_tokenize(ctx->vocab, text, add_bos);

  if (n_max_tokens < (int)res.size()) {
    fprintf(stderr, "%s: too many tokens\n", __func__);
    return -((int)res.size());
  }

  for (size_t i = 0; i < res.size(); i++) {
    tokens[i] = res[i];
  }

  return res.size();
}

int falcon_n_vocab(const struct falcon_context *ctx) {
  return ctx->vocab.id_to_token.size();
}

int falcon_n_ctx(const struct falcon_context *ctx) {
  return ctx->model.hparams.n_ctx;
}

int falcon_n_embd(const struct falcon_context *ctx) {
  return ctx->model.hparams.n_embd;
}

int falcon_get_vocab(const struct falcon_context *ctx, const char **strings,
                     float *scores, int capacity) {
  int n = std::min(capacity, (int)ctx->vocab.id_to_token.size());
  for (int i = 0; i < n; ++i) {
    strings[i] = ctx->vocab.id_to_token[i].tok.c_str();
    scores[i] = ctx->vocab.id_to_token[i].score;
  }
  return n;
}

float *falcon_get_logits(struct falcon_context *ctx) {
  return ctx->logits.data();
}

float *falcon_get_embeddings(struct falcon_context *ctx) {
  return ctx->embedding.data();
}

const char *falcon_token_to_str(const struct falcon_context *ctx,
                                llama_token token) {
  if (token >= falcon_n_vocab(ctx)) {
    return nullptr;
  }

  return ctx->vocab.id_to_token[token].tok.c_str();
}

llama_token falcon_token_bos() { return 11; }

llama_token falcon_token_eos() { return 11; }

llama_token falcon_token_nl() { return 193; }

llama_token falcon_token_cr() { return 195; }

void falcon_print_timings(struct falcon_context *ctx) {
  const int64_t t_end_us = ggml_time_us();

  const int32_t n_sample = std::max(1, ctx->n_sample);
  const int32_t n_eval = std::max(1, ctx->n_eval);
  const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__,
          ctx->t_load_us / 1000.0);
  fprintf(stderr,
          "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token)\n",
          __func__, 1e-3 * ctx->t_sample_us, n_sample,
          1e-3 * ctx->t_sample_us / n_sample);
  fprintf(stderr,
          "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n",
          __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval,
          1e-3 * ctx->t_p_eval_us / n_p_eval);
  fprintf(stderr,
          "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token)\n",
          __func__, 1e-3 * ctx->t_eval_us, n_eval,
          1e-3 * ctx->t_eval_us / n_eval);
  fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__,
          (t_end_us - ctx->t_start_us) / 1000.0);
}

void falcon_reset_timings(struct falcon_context *ctx) {
  ctx->t_start_us = ggml_time_us();
  ctx->t_sample_us = ctx->n_sample = 0;
  ctx->t_eval_us = ctx->n_eval = 0;
  ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char *falcon_print_system_info(void) {
  static std::string s;

  s = "";
  s += "AVX = " + std::to_string(ggml_cpu_has_avx()) + " | ";
  s += "AVX2 = " + std::to_string(ggml_cpu_has_avx2()) + " | ";
  s += "AVX512 = " + std::to_string(ggml_cpu_has_avx512()) + " | ";
  s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
  s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
  s += "FMA = " + std::to_string(ggml_cpu_has_fma()) + " | ";
  s += "NEON = " + std::to_string(ggml_cpu_has_neon()) + " | ";
  s += "ARM_FMA = " + std::to_string(ggml_cpu_has_arm_fma()) + " | ";
  s += "F16C = " + std::to_string(ggml_cpu_has_f16c()) + " | ";
  s += "FP16_VA = " + std::to_string(ggml_cpu_has_fp16_va()) + " | ";
  s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
  s += "BLAS = " + std::to_string(ggml_cpu_has_blas()) + " | ";
  s += "SSE3 = " + std::to_string(ggml_cpu_has_sse3()) + " | ";
  s += "VSX = " + std::to_string(ggml_cpu_has_vsx()) + " | ";

  return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ggml_tensor *>>
    &llama_internal_get_tensor_map(struct falcon_context *ctx) {
  return ctx->model.tensors_by_name;
}
