/*
 * libfalcon.cpp - core functions for inference and loading of ggcc type falcon
 * 7B and 40B models https://github.com/cmp-nct/ggllm.cpp MIT licensed,
 * contributions welcome
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "cmpnct_unicode.h"
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
// #include <locale>

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
static const char* unused0 = FINETUNE_NAME[0];

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

static const std::map<falcon_e_model, size_t>& FALCON_MEM_REQ_SCRATCH0() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 32ull * MB},
      {FALCON_40B, 32ull * MB},
  };
  return k_sizes;
}

static const std::map<falcon_e_model, size_t>& FALCON_MEM_REQ_SCRATCH1() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 32ull * MB},
      {FALCON_40B, 32ull * MB},
  };
  return k_sizes;
}
// batch mode requires growingly large tensors
static std::pair<size_t, size_t> FALCON_MEM_REQ_EVAL_BATCH(
    falcon_e_model model, int32_t n_batch, int32_t n_ctx_prompt) {
  size_t batch_scratch0 = 0;
  size_t batch_scratch1 = 0;
  // currently does not calculate if batch reduced at last passt (so assumes ctx
  // is a clean multiple of batch) that would cause higher memory allocation
  // than needed todo: calculate the multipliers based on actual tensor sizes
  double a = 0;
  double d = 0;
  double lin_a = n_ctx_prompt * n_batch;
  switch (model) {
    case FALCON_7B:
      a = 0.00029706;
      d = 92;
      batch_scratch0 = (size_t)(lin_a * a + d) * MB;
      batch_scratch1 = (size_t)145752 * (size_t)n_batch + 8 * MB;
      break;

    case FALCON_40B:
      a = 0.00065;
      d = 118;
      batch_scratch0 = (size_t)(lin_a * a + d) * MB;
      batch_scratch1 =
          (size_t)262144ull * (size_t)n_batch + 8 * MB;  // 25mb clean/100 batch
      break;
    default:
      break;
  }

  return std::make_pair(batch_scratch0, batch_scratch1);
}

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// todo: check if this is actually needed - most happens in scratch 0
static const std::map<falcon_e_model, size_t>& FALCON_MEM_REQ_EVAL() {
  static std::map<falcon_e_model, size_t> k_sizes = {
      {FALCON_7B, 160ull * MB},
      {FALCON_40B,
       256ull * MB},  // for full offload matmul GPU this is oversized
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
  int32_t n_bpe_merges =
      64784;  // in binary starting with FALCON_FILE_VERSION_GGCC_V1
  enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;

  bool operator!=(const falcon_hparams& other) const {
    return static_cast<bool>(memcmp(this, &other, sizeof(falcon_hparams)));
  }
};

static size_t FALCON_MEM_REQ_KV_SELF(const falcon_hparams& hparams,
                                     ggml_type wtype, int32_t n_ctx) {
  const int n_head_kv = hparams.n_head_kv;
  const int head_dim = hparams.n_embd / hparams.n_head;
  const int n_layer = hparams.n_layer;

  const int64_t ne = n_head_kv * head_dim * n_layer * n_ctx;
#ifdef FALCON_NO_KV_UPGRADE
  return ggml_tensor_overhead() * 2 +
         2u * (ggml_tensor_overhead() + ne * ggml_type_size(wtype));
#else
  return ggml_tensor_overhead() +
         3u * (ggml_tensor_overhead() + ne * ggml_type_size(wtype));
#endif
}

struct falcon_layer {
  // normalization
  struct ggml_tensor* input_layernorm;
  struct ggml_tensor* input_layernorm_b;
  struct ggml_tensor* attention_norm;    // Falcon-40B only
  struct ggml_tensor* attention_norm_b;  // Falcon-40B only

  // attention
  struct ggml_tensor* query_key_value;
  struct ggml_tensor* wo;

  // ff
  struct ggml_tensor* ffn_up;
  struct ggml_tensor* ffn_down;
};

struct falcon_kv_cache {
  struct ggml_tensor* k;
  struct ggml_tensor* v;  // only used with FALCON_NO_KV_UPGRADE
  struct ggml_tensor* v_a;
  struct ggml_tensor* v_b;

  typedef enum v_buftype { V_A, V_B } v_buftype_t;
  v_buftype_t v_current = V_A;

  struct ggml_context* ctx = NULL;

  llama_ggml::llama_ctx_buffer buf;

  int n;  // number of tokens currently in the cache

  ~falcon_kv_cache() {
    if (ctx) {
      ggml_free(ctx);
    }
  }
};

#if 1

std::string replaceAll(std::string str, const std::string& from,
                       const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos +=
        to.length();  // Handles case where 'to' is a substring of 'from'
  }
  return str;
}
struct TrieNode {
  std::map<char, TrieNode*> map;
  int32_t Id = -1;
};
struct Trie {
  TrieNode* root;

  Trie() : root(new TrieNode()) {}
  ~Trie() {
    if (root) deleteTrie(root);
  }
  // Move constructor
  Trie(Trie&& other) noexcept : root(other.root) { other.root = nullptr; }

  // Move assignment operator
  Trie& operator=(Trie&& other) noexcept {
    if (this != &other) {
      if (root) deleteTrie(root);
      root = other.root;
      other.root = nullptr;
    }
    return *this;
  }

  void insert(const std::string& token, int32_t Id) {
    TrieNode* current = root;
    for (auto ch : token) {
      if (current->map.find(ch) == current->map.end())
        current->map[ch] = new TrieNode();
      current = current->map[ch];
    }
    current->Id = Id;
  }

  void reset() {
    deleteTrie(root);
    root = new TrieNode();
  }

 private:
  void deleteTrie(TrieNode* node) {
    for (auto& it : node->map) {
      deleteTrie(it.second);
    }
    delete node;
  }
};
struct falcon_vocab {
  using id = int32_t;
  using token = std::string;
  std::map<std::string, uint32_t>
      max_token_length;  // max length, for each 2byte prefix
  // std::unordered_map<std::pair<std::string, std::string>, int> bpe_ranks;
  std::map<std::pair<std::string, std::string>, int> bpe_ranks;
  std::vector<std::pair<std::string, std::string>> bpe_merges;
  std::map<std::string, int> special_tokens;

  struct token_score {
    token tok;
    float score;
  };

  std::unordered_map<token, id> token_to_id;
  std::vector<token_score> id_to_token;
  Trie trie;  // highspeed access to tokens by prefix tree

  // populate trie from map
  void populate_trie_from_map() {
    trie.reset();
    for (const auto& pair : token_to_id) {
      trie.insert(pair.first, pair.second);
      if (pair.first.size() >= 2) {
        std::string prefix = pair.first.substr(0, 2);
        max_token_length[prefix] =
            std::max(max_token_length[prefix], (uint32_t)pair.first.size());
      }
    }
  }
  // populate token ranks map
  int populate_bpe_ranks(
      std::vector<std::pair<std::string, std::string>> bpe_merges_) {
    for (int i = 0; i < (int)bpe_merges_.size(); i++) {
      bpe_ranks.emplace(bpe_merges_[i], i);
    }
    bpe_merges = bpe_merges_;

    // populate special tokens too (0-11 and if available 65024++)
    for (int i = 0; i < 12; i++) {
      special_tokens[id_to_token[i].tok] = i;
    }
    for (int i = 65024; i < (int)id_to_token.size(); i++) {
      special_tokens[id_to_token[i].tok] = i;
    }
    // token_to_id["</s>"] = 11; // bugfix for TII instruct training (blocks
    // stopwords) special_tokens["</s>"] = 11; // bugfix for TII instruct
    // training (blocks stopwords)

    return bpe_merges_.size();
  }
  // Trim whitespace characters from the beginning and end of the string
  void trim(std::string& str) {
    // Remove whitespace characters from the beginning of the string
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) {
                return !std::isspace(ch);
              }));

    // Remove whitespace characters from the end of the string
    str.erase(std::find_if(str.rbegin(), str.rend(),
                           [](int ch) { return !std::isspace(ch); })
                  .base(),
              str.end());
  }
  // requires the standard HF type tokenizer.json (pretty printed)
  std::vector<std::pair<std::string, std::string>> parse_json_to_bpe_merges(
      const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
      fprintf(stderr, "Error: could not open file %s\n", filename.c_str());
      return std::vector<
          std::pair<std::string, std::string>>();  // return empty vector
    }

    std::string line;
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    bool isMergesSection = false;
    int count = 0;

    while (std::getline(file, line)) {
      // Trim the line
      line.erase(0, line.find_first_not_of(" \t"));
      line.erase(line.find_last_not_of(" \t") + 1);
      trim(line);

      if (line == "\"merges\": [") {
        isMergesSection = true;
        continue;
      }

      if (isMergesSection) {
        static char allowedSymbols[] = {' ', '\t', '\n', '\r', ']', ',', ';'};

        std::vector<char> bytes(
            line.begin(), line.end());  // Convert the string to vector of bytes
        if (bytes.size() <= 8) {
          bool is_finished = true;
          for (const char& byte : bytes) {
            if (std::find(std::begin(allowedSymbols), std::end(allowedSymbols),
                          byte) == std::end(allowedSymbols)) {
              is_finished = false;
              break;
            }
          }
          if (is_finished) break;
        }

        // Remove the leading and trailing quotes
        if (line.size() < 2) {
          fprintf(stderr, "Error: Invalid line format in file %s\n",
                  filename.c_str());
          fprintf(stderr, "Line: %s\n", line.c_str());
          return std::vector<
              std::pair<std::string, std::string>>();  // return empty vector
        }
        size_t pos = line.find('"');
        if (pos != std::string::npos) {
          line.erase(0, pos + 1);
        }
        pos = line.rfind('"');
        if (pos != std::string::npos) {
          line.erase(pos);
        }

        std::istringstream iss(line);
        std::string first, second;
        pos = line.find(' ', 1);  // Start the search from the second character
        if (pos != std::string::npos) {
          first = line.substr(0, pos);
          second = line.substr(pos + 1);
        }
        for (auto& str : {std::ref(first), std::ref(second)}) {
          size_t pos = 0;
          while ((pos = str.get().find("\\\\", pos)) != std::string::npos) {
            str.get().replace(pos, 2, "\\");
            pos += 1;
          }
          pos = 0;
          while ((pos = str.get().find("\\\"", pos)) != std::string::npos) {
            str.get().replace(pos, 2, "\"");
            pos += 1;
          }
        }
        bpe_merges.push_back(std::make_pair(first, second));
        count++;
      }
    }

    if (bpe_merges.empty()) {
      fprintf(stderr, "Error: could not parse file %s\n", filename.c_str());
    }

    return bpe_merges;
  }

  // get max token length available for a prefix of 2 bytes (string at least 2
  // bytes long)
  int get_max_token_length(const std::string& string) const {
    if (string.size() < 2) return -1;
    std::string prefix = string.substr(0, 2);
    if (max_token_length.find(prefix) == max_token_length.end()) return 0;
    return max_token_length.at(prefix);
  }

  // function to find if two tokens match in bpe_rank, return rank or -1
  int find_bpe_rank(const std::string& token1,
                    const std::string& token2) const {
    std::string left_token = token1;
    std::string right_token = token2;
    left_token = replaceAll(left_token, " ", "Ġ");
    left_token = replaceAll(left_token, "\n", "Ċ");
    right_token = replaceAll(right_token, " ", "Ġ");
    right_token = replaceAll(right_token, "\n", "Ċ");

    auto it = bpe_ranks.find(std::make_pair(left_token, right_token));
    if (it == bpe_ranks.end()) return -1;
    return it->second;
  }

  std::pair<falcon_vocab::id, std::string> find_longest_match(
      const std::string& snippet) const {
    TrieNode* current = trie.root;
    falcon_vocab::id last_matched_id = -1;
    std::string last_matched_token = "";
    std::string current_token = "";
    for (auto ch : snippet) {
      if (current->map.find(ch) == current->map.end()) {
        break;
      }
      current = current->map[ch];
      current_token += ch;
      if (current->Id != -1) {
        last_matched_id = current->Id;
        last_matched_token = current_token;
      }
    }
    return {last_matched_id, last_matched_token};
  }
};
#endif
struct falcon_model {
  falcon_e_model type = FALCON_UNKNOWN;

  falcon_hparams hparams;

  falcon_vocab vocab;

  struct ggml_tensor* tok_embeddings;
  struct ggml_tensor* output_norm;
  struct ggml_tensor* output_norm_b;
  struct ggml_tensor* lm_head;

  std::vector<falcon_layer> layers;

  int n_gpu_layers;
  int i_gpu_start;
  int i_gpu_last;

  // context
  struct ggml_context* ctx = NULL;
  std::map<std::string, struct ggml_tensor*> tensors;

  // key + value cache for the self attention
  // TODO: move to llama_state
  struct falcon_kv_cache kv_self;

  // the model memory buffer
  llama_ggml::llama_ctx_buffer buf;

  // model memory mapped file
  std::unique_ptr<llama_ggml::llama_mmap> mapping;

  // objects representing data potentially being locked in memory
  llama_mlock mlock_buf;
  llama_mlock mlock_mmap;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor*>> tensors_by_name;

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
  falcon_context(falcon_model& model, falcon_vocab& vocab)
      : model(model), vocab(vocab) {}
  std::string context_name = "default";
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

  falcon_model& model;
  falcon_vocab& vocab;

  size_t mem_per_token = 0;

  // decode output (2-dimensional array: [n_tokens][n_vocab])
  std::vector<float> logits;
  bool logits_all = false;

  // input embedding (1-dimensional array: [n_embd])
  std::vector<float> embedding;

  // memory buffers used to evaluate the model
  // TODO: move in llama_state
  llama_ggml::llama_ctx_buffer buf_compute;
  llama_ggml::llama_ctx_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

#ifdef GGML_USE_METAL
  ggml_metal_context* ctx_metal = NULL;
#endif

  int buf_last = 0;
  size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = {0};

  void use_buf(struct ggml_context* ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
    size_t last_size = 0;

    if (i == -1) {
      last_size = ggml_set_scratch(ctx, {
                                            0,
                                            0,
                                            nullptr,
                                        });
    } else {
      auto& buf = buf_scratch[i];
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

struct falcon_load_tensor_shard {
  std::vector<uint32_t> ne;
  size_t size;
  enum ggml_type type;
  size_t file_idx;
  size_t file_off;

  void calc_size() { size = llama_ggml::llama_calc_tensor_size(ne, type); }
};

enum falcon_split_type { SPLIT_NONE, SPLIT_BY_COLUMNS, SPLIT_BY_ROWS };

struct falcon_load_tensor {
  std::vector<falcon_load_tensor_shard> shards;

  std::string name;
  enum ggml_type type = GGML_TYPE_F32;
  falcon_split_type split_type = SPLIT_NONE;
  std::vector<uint32_t> ne;
  size_t size;
  struct ggml_tensor* ggml_tensor = NULL;
  uint8_t* data;

  falcon_load_tensor(const std::string& name) : name(name) {}

  void calc_all() {
    calc_type();
    calc_split_type();
    calc_ne();
    calc_size();
  }

  void calc_type() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.type != first_shard.type) {
        throw std::runtime_error(
            format("inconsistent tensor shard type in '%s'", name.c_str()));
      }
    }
    type = first_shard.type;
  }

  void calc_split_type() {
    if (shards.at(0).ne.size() ==
            1 ||               // 1D tensors are just duplicated in every file
        shards.size() == 1) {  // only one file?
      split_type = SPLIT_NONE;
    } else if (name.find("tok_embeddings.") == 0 ||
               name.find(".attention.wo.weight") != std::string::npos ||
               name.find(".feed_forward.w2.weight") != std::string::npos) {
      split_type = SPLIT_BY_COLUMNS;
    } else {
      split_type = SPLIT_BY_ROWS;
    }
  }

  void calc_ne() {
    const auto& first_shard = shards.at(0);
    for (const auto& shard : shards) {
      if (shard.ne != first_shard.ne) {
        throw std::runtime_error(format(
            "inconsistent tensor shard shape in '%s': first was %s, other was "
            "%s",
            name.c_str(),
            llama_ggml::llama_format_tensor_shape(first_shard.ne).c_str(),
            llama_ggml::llama_format_tensor_shape(shard.ne).c_str()));
      }
    }
    ne = first_shard.ne;
    LLAMA_ASSERT(shards.size() <= UINT32_MAX);
    uint32_t n_shards = (uint32_t)shards.size();
    switch (split_type) {
      case SPLIT_NONE:
        ne = first_shard.ne;
        break;
      case SPLIT_BY_COLUMNS:
        ne = {llama_ggml::checked_mul<uint32_t>(first_shard.ne[0], n_shards),
              first_shard.ne[1]};
        break;
      case SPLIT_BY_ROWS:
        ne = {first_shard.ne[0],
              llama_ggml::checked_mul<uint32_t>(first_shard.ne[1], n_shards)};
        break;
    }
  }

  void calc_size() { size = llama_ggml::llama_calc_tensor_size(ne, type); }
};

struct falcon_load_tensors_map {
  // tensors is kept in a separate vector to preserve file order
  std::vector<falcon_load_tensor> tensors;
  std::unordered_map<std::string, size_t> name_to_idx;
};

enum falcon_file_version {
  FALCON_FILE_VERSION_GGML,  // ftype incompatible when using the current falcon
                             // converter, remainder is ggml format
  FALCON_FILE_VERSION_GGMF_V1,  // added version field and scores in vocab
  FALCON_FILE_VERSION_GGJT_V1,  // added padding
  FALCON_FILE_VERSION_GGJT_V2,  // changed quantization format
  FALCON_FILE_VERSION_GGJT_V3,  // changed Q4 and Q8 quantization format
  RESERVED_1,
  RESERVED_2,
  RESERVED_3,
  RESERVED_4,
  RESERVED_5,
  FALCON_FILE_VERSION_GGCC_V1  // for new falcon tokenizer

};

struct falcon_file_loader {
  llama_ggml::llama_file file;
  falcon_file_version file_version;
  falcon_hparams hparams;
  falcon_vocab vocab;
  const char* fname;

  falcon_file_loader(const char* fname, size_t file_idx,
                     falcon_load_tensors_map& tensors_map)
      : file(fname, "rb") {
    this->fname = fname;
    read_magic();
    read_hparams();
    read_vocab();
    read_tensor_metadata(file_idx, tensors_map);
  }
  void read_magic() {
    uint32_t magic = file.read_u32();

    if (magic == LLAMA_FILE_MAGIC_GGML) {
      file_version = FALCON_FILE_VERSION_GGML;
      return;
    }

    uint32_t version = file.read_u32();
    switch (magic) {
      case LLAMA_FILE_MAGIC_GGMF:
        switch (version) {
          case 1:
            file_version = FALCON_FILE_VERSION_GGMF_V1;
            return;
        }
        break;
      case LLAMA_FILE_MAGIC_GGJT:
        switch (version) {
          case 1:
            file_version = FALCON_FILE_VERSION_GGJT_V1;
            return;
          case 2:
            file_version = FALCON_FILE_VERSION_GGJT_V2;
            return;
          case 3:
            file_version = FALCON_FILE_VERSION_GGJT_V3;
            return;
            break;
        }
        break;
      case FALCON_FILE_MAGIC_GGCC:
        switch (version) {
          // we start at 10 to avoid confusion with the old GGJT format
          case 10:
            file_version = FALCON_FILE_VERSION_GGCC_V1;
            return;
            break;
        }
        break;
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
    if (file_version == FALCON_FILE_VERSION_GGML) {
      int32_t ftype = file.read_u32();
      const int32_t qntvr = ftype / GGML_QNT_VERSION_FACTOR;
      hparams.ftype = (enum llama_ftype)(qntvr);
    } else {
      hparams.ftype = (enum llama_ftype)file.read_u32();
    }
    if (file_version >= FALCON_FILE_VERSION_GGCC_V1) {
      hparams.n_bpe_merges = file.read_u32();
    }
  }
  void read_vocab() {
    vocab.id_to_token.resize(hparams.n_vocab);
    for (uint32_t i = 0; i < (uint32_t)hparams.n_vocab; i++) {
      uint32_t len = file.read_u32();
      std::string word = file.read_string(len);
      // the vocab is not identical to HF, the whitespace prefix is an actual
      // #20 space
      float score = 0.0f;  // flacon does not have scores in vocab, scores are a
                           // sentencepiece addition
      if (file_version >= FALCON_FILE_VERSION_GGMF_V1) {
        file.read_raw(&score, sizeof(score));
      }
      vocab.token_to_id[word] = i;

      auto& tok_score = vocab.id_to_token[i];
      tok_score.tok = std::move(word);
      tok_score.score = score;
    }
    if (file_version >= FALCON_FILE_VERSION_GGJT_V3 &&
        hparams.n_vocab == 65025 && vocab.id_to_token[65024].tok == "[PAD]") {
      // wizard hack - shaving off one token
      vocab.id_to_token.resize(65024);
      vocab.token_to_id.erase("[PAD]");
      hparams.n_vocab = 65024;
      // this needs a followup of the tensors itself, quality of the model
      // affected needs to be tested
    }
    if (file_version >= FALCON_FILE_VERSION_GGCC_V1) {
      // similar to vocab the merges are read
      std::vector<std::pair<std::string, std::string>> bpe_merges;
      int32_t num_bpe_merges = file.read_u32();
      for (int i = 0; i < num_bpe_merges; i++) {
        uint32_t len1 = file.read_u32();
        std::string word1 = file.read_string(len1);
        uint32_t len2 = file.read_u32();
        std::string word2 = file.read_string(len2);
        bpe_merges.push_back(std::make_pair(word1, word2));
      }
      vocab.populate_bpe_ranks(bpe_merges);
    } else {
      // same path as model file, we need to cut the filename off from fname:
      fprintf(stderr,
              "falcon.cpp: fallback for old file format. Loading BPE merges "
              "from tokenizer.json\n");
      std::string fname_str(fname);
      size_t last_slash_idx = fname_str.find_last_of("\\/");
      std::string parent_path = fname_str.substr(0, last_slash_idx);
      std::string tokenizer_json_path = parent_path + "/tokenizer.json";
      auto merges = vocab.parse_json_to_bpe_merges(tokenizer_json_path);
      if (merges.empty()) {
        fprintf(stderr,
                "falcon.cpp: error: old file format. Place json data in "
                "directory: %s\n",
                tokenizer_json_path.c_str());
#if defined(GGML_USE_CUBLAS)
        // while (!ggml_init_cublas(true))
        //   std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif
        exit(1);
      }
      int num_bpe_merges = vocab.populate_bpe_ranks(merges);
      if (num_bpe_merges == 0) {
        fprintf(stderr,
                "falcon.cpp: error: old file format, no valid BPE merges found "
                "in %s\n",
                tokenizer_json_path.c_str());
#if defined(GGML_USE_CUBLAS)
        // while (!ggml_init_cublas(true))
        //   std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif
        exit(1);
      }
      hparams.n_bpe_merges = num_bpe_merges;
    }

#if 1
    vocab.populate_trie_from_map();
#endif
  }
  void read_tensor_metadata(size_t file_idx,
                            falcon_load_tensors_map& tensors_map) {
    while (file.tell() < file.size) {
      falcon_load_tensor_shard shard;
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

      if (file_version >= FALCON_FILE_VERSION_GGJT_V1) {
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
  llama_ggml::llama_file file;
  falcon_file_loader* any_file_loader;
  falcon_file_saver(const char* fname, falcon_file_loader* any_file_loader,
                    enum llama_ftype new_ftype)
      : file(fname, "wb"), any_file_loader(any_file_loader) {
    any_file_loader->file_version = FALCON_FILE_VERSION;
    fprintf(stderr, "falcon.cpp: saving model to %s\n", fname);
    write_magic();
    write_hparams(new_ftype);
    write_vocab();
  }
  void write_magic() {
    file.write_u32(FALCON_FILE_MAGIC);    // magic
    file.write_u32(FALCON_FILE_VERSION);  // version
  }
  void write_hparams(enum llama_ftype new_ftype) {
    const falcon_hparams& hparams = any_file_loader->hparams;
    file.write_u32(hparams.n_vocab);
    file.write_u32(hparams.n_embd);
    file.write_u32(hparams.n_head);
    file.write_u32(hparams.n_head_kv);
    file.write_u32(hparams.n_layer);
    file.write_u32(hparams.n_falcon_type);
    file.write_u32(new_ftype);
    if (any_file_loader->file_version >= FALCON_FILE_VERSION_GGCC_V1) {
      file.write_u32(hparams.n_bpe_merges);
    }
  }
  void write_vocab() {
    if (any_file_loader->file_version == FALCON_FILE_VERSION_GGML) {
      // fprintf(stderr, "falcon.cpp: WARNING: input is an old file that doesn't
      // have scores; will add dummy scores\n");
    }
    uint32_t n_vocab = any_file_loader->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
      const auto& token = any_file_loader->vocab.id_to_token.at(i);
      file.write_u32((uint32_t)token.tok.size());
      file.write_raw(token.tok.data(), token.tok.size());
      file.write_raw(&token.score, sizeof(token.score));
    }
    if (any_file_loader->file_version >= FALCON_FILE_VERSION_GGCC_V1) {
      file.write_u32(any_file_loader->hparams.n_bpe_merges);
      for (int32_t i = 0; i < any_file_loader->hparams.n_bpe_merges; i++) {
        const auto& bpe_merge = any_file_loader->vocab.bpe_merges.at(i);
        file.write_u32((uint32_t)bpe_merge.first.size());
        file.write_raw(bpe_merge.first.data(), bpe_merge.first.size());
        file.write_u32((uint32_t)bpe_merge.second.size());
        file.write_raw(bpe_merge.second.data(), bpe_merge.second.size());
      }
    }
  }
  void write_tensor(falcon_load_tensor& tensor, enum ggml_type new_type,
                    const void* new_data, size_t new_size) {
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
    LLAMA_ASSERT(new_size ==
                 llama_ggml::llama_calc_tensor_size(tensor.ne, new_type));
    file.write_raw(new_data, new_size);
  }
};

struct falcon_model_loader {
  std::vector<std::unique_ptr<falcon_file_loader>> file_loaders;
  falcon_load_tensors_map tensors_map;
  bool use_mmap;
  size_t num_ggml_tensors_created = 0;
  struct ggml_context* ggml_ctx = NULL;
  std::unique_ptr<llama_ggml::llama_mmap> mapping;

  falcon_model_loader(const std::string& fname_base, bool use_mmap,
                      bool vocab_only) {
    auto* first_file =
        new falcon_file_loader(fname_base.c_str(), 0, tensors_map);
    file_loaders.emplace_back(first_file);
    uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
    for (uint32_t i = 1; i < n_parts; i++) {
      std::string fname = fname_base + "." + std::to_string(i);
      auto* ith_file = new falcon_file_loader(fname.c_str(), i, tensors_map);
      file_loaders.emplace_back(ith_file);
      if (ith_file->hparams != first_file->hparams) {
        throw std::runtime_error(
            format("falcon.cpp: hparams inconsistent between files"));
      }
    }
    if (!llama_ggml::llama_mmap::SUPPORTED) {
      use_mmap = false;
    }
    if (use_mmap && alignment_prevents_mmap()) {
      fprintf(stderr,
              "falcon.cpp: can't use mmap because tensors are not aligned; "
              "convert to new format to avoid this\n");
      use_mmap = false;
    }
    this->use_mmap = use_mmap;
    for (falcon_load_tensor& lt : tensors_map.tensors) {
      lt.calc_all();
    }
  }

  bool alignment_prevents_mmap() {
    for (const falcon_load_tensor& lt : tensors_map.tensors) {
      for (const falcon_load_tensor_shard& shard : lt.shards) {
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
      throw std::runtime_error(std::string("missing word_embeddings.weight"));
    }
    const falcon_load_tensor& lt = tensors_map.tensors.at(it->second);
    return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
  }

  void calc_sizes(size_t* ctx_size_p, size_t* mmapped_size_p) const {
    *ctx_size_p = *mmapped_size_p = 0;
    for (const falcon_load_tensor& lt : tensors_map.tensors) {
      *ctx_size_p += ggml_tensor_overhead();
      *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
    }
  }

  struct ggml_tensor* get_tensor(const std::string& name,
                                 const std::vector<uint32_t>& ne,
                                 ggml_backend backend) {
    auto it = tensors_map.name_to_idx.find(name);
    if (it == tensors_map.name_to_idx.end()) {
      throw std::runtime_error(std::runtime_error(format(
          "falcon.cpp: tensor '%s' is missing from model", name.c_str())));
    }
    falcon_load_tensor& lt = tensors_map.tensors.at(it->second);
    // special case - wizard padding token - todo: move into quantizer only
    // (only with mmap versions)
    if (!use_mmap || lt.ne.size() != 2 || ne.size() != 2 ||
        !((ne[0] == 8192 && ne[1] == 65024 && lt.ne[0] == 8192 &&
           lt.ne[1] == 65025) ||
          (ne[0] == 4544 && ne[1] == 65024 && lt.ne[0] == 4544 &&
           lt.ne[1] == 65025)))
      if (lt.ne != ne) {
        throw std::runtime_error(format(
            "falcon.cpp: tensor '%s' has wrong shape; expected %s, got %s",
            name.c_str(), llama_ggml::llama_format_tensor_shape(ne).c_str(),
            llama_ggml::llama_format_tensor_shape(lt.ne).c_str()));
      }
    // printf("Tensor: %-70s %s\n", name.c_str(),
    // llama_ggml::llama_format_tensor_shape(lt.ne).c_str());

    return get_tensor_for(lt, backend);
  }

  struct ggml_tensor* get_tensor_for(falcon_load_tensor& lt,
                                     ggml_backend backend) {
    struct ggml_tensor* tensor;
    if (backend != GGML_BACKEND_CPU) {
      ggml_set_no_alloc(ggml_ctx, true);
    }
    if (lt.ne.size() == 2) {
      if (use_mmap && lt.ne[0] == 8192 && lt.ne[1] == 65025 &&
          (lt.name.find("transformer.word_embeddings.weight") == 0 ||
           lt.name.find("lm_head.weight") == 0)) {
        // TODO: evaluate if we lose perplexity from the changed shape
        tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, 8192, 65024);
      } else if (use_mmap && lt.ne[0] == 4544 && lt.ne[1] == 65025 &&
                 (lt.name.find("transformer.word_embeddings.weight") == 0 ||
                  lt.name.find("lm_head.weight") == 0)) {
        // TODO: evaluate if we lose perplexity from the changed shape
        tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, 4544, 65024);
      } else {
        tensor =
            ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
      }
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

  void verify_correct_load() const {
    if (num_ggml_tensors_created != tensors_map.tensors.size()) {
      throw std::runtime_error(
          std::string("falcon.cpp: file contained more tensors than expected"));
    }
  }

  void load_all_data(falcon_progress_callback progress_callback,
                     void* progress_callback_user_data, llama_mlock* lmlock) {
    size_t data_size = 0;
    size_t prefetch_size = 0;
    size_t lock_size = 0;
    for (const falcon_load_tensor& lt : tensors_map.tensors) {
      data_size += lt.size;
      if (lt.ggml_tensor->backend == GGML_BACKEND_CPU) {
        prefetch_size += lt.size;
      }
    }

    if (use_mmap) {
      mapping.reset(
          new llama_ggml::llama_mmap(&file_loaders.at(0)->file, prefetch_size));
      if (lmlock) {
        lmlock->init(mapping->addr);
      }
    }

    size_t done_size = 0;
    for (falcon_load_tensor& lt : tensors_map.tensors) {
      if (progress_callback) {
        const char* status = "";
        if (lt.ggml_tensor->backend == GGML_BACKEND_CPU)
          status = "Loading tensor (CPU)";
        else if (lt.ggml_tensor->backend == GGML_BACKEND_GPU)
          status = "Loading tensor (GPU-Main)";
        else if (lt.ggml_tensor->backend == GGML_BACKEND_GPU_SPLIT)
          status = "Loading tensor (GPU-Split)";

        progress_callback((float)done_size / data_size,
                          progress_callback_user_data, status);
      }
      LLAMA_ASSERT(lt.ggml_tensor);  // unused tensors should have been caught
                                     // by load_data already
      lt.data = (uint8_t*)lt.ggml_tensor->data;

      // allocate temp buffer if not using mmap
      if (!use_mmap && lt.data == NULL) {
        GGML_ASSERT(lt.ggml_tensor->backend != GGML_BACKEND_CPU);
        lt.data = (uint8_t*)malloc(ggml_nbytes(lt.ggml_tensor));
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
          // there is no downside assigning the host pointer as long as we've
          // the RAM to hold the mmap region
          lt.ggml_tensor->data = lt.data;
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

  void load_data_for(falcon_load_tensor& lt) {
    if (use_mmap) {
      LLAMA_ASSERT(lt.shards.size() == 1);
      lt.data = (uint8_t*)mapping->addr + lt.shards.at(0).file_off;
    } else if (lt.split_type == SPLIT_NONE) {
      llama_ggml::llama_file& file =
          file_loaders.at(lt.shards.at(0).file_idx)->file;
      file.seek(lt.shards.at(0).file_off, SEEK_SET);
      file.read_raw(lt.data, lt.size);
    } else if (lt.split_type == SPLIT_BY_ROWS) {
      size_t offset = 0;
      for (falcon_load_tensor_shard& shard : lt.shards) {
        llama_ggml::llama_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        file.read_raw(lt.data + offset, shard.size);
        offset += shard.size;
      }
      LLAMA_ASSERT(offset == lt.size);
    } else if (lt.split_type == SPLIT_BY_COLUMNS) {
      // Let's load the data into temporary buffers to ensure the OS performs
      // large loads.
      std::vector<llama_ggml::llama_buffer> tmp_bufs(lt.shards.size());
      for (size_t i = 0; i < lt.shards.size(); i++) {
        falcon_load_tensor_shard& shard = lt.shards.at(i);
        llama_ggml::llama_file& file = file_loaders.at(shard.file_idx)->file;
        file.seek(shard.file_off, SEEK_SET);
        tmp_bufs.at(i).resize(shard.size);
        file.read_raw(tmp_bufs.at(i).addr, shard.size);
      }
      // Then reshape.
      size_t num_rows = lt.ne.at(1);
      size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
      size_t out_offset = 0;
      for (size_t row = 0; row < num_rows; row++) {
        for (llama_ggml::llama_buffer& tmp_buf : tmp_bufs) {
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

  static void print_checksum(falcon_load_tensor& lt) {
    uint32_t sum = 0;
    for (size_t i = 0; i < lt.size; i++) {
      uint8_t byte = lt.data[i];
      sum = byte + (sum << 6) + (sum << 16) - sum;  // sdbm hash
    }
    fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
            llama_ggml::llama_format_tensor_shape(lt.ne).c_str(), lt.size);
  }
};

//
// kv cache
//

static bool kv_cache_init(const struct falcon_hparams& hparams,
                          struct falcon_kv_cache& cache, ggml_type wtype,
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
#ifdef FALCON_NO_KV_UPGRADE
  cache.v = ggml_new_tensor_1d(cache.ctx, wtype,
                               n_elements);  // only used as reference now
#else
  cache.v =
      ggml_new_tensor_1d(cache.ctx, wtype, 0);  // only used as reference now
#endif
  cache.v_a = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
  cache.v_b = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
  ggml_set_name(cache.k, "cache_k");
  ggml_set_name(cache.v, "cache_v");
  ggml_set_name(cache.v_a, "cache_v_a");
  ggml_set_name(cache.v_b, "cache_v_b");

  (void)n_gpu_layers;
#ifdef GGML_USE_CUBLAS
  // TODO : gpu kv offloading not implemented
  if (n_gpu_layers > n_layer + 1) {
    ggml_cuda_assign_buffers_no_scratch(cache.k);
    ggml_cuda_assign_buffers_no_scratch(cache.v);
    ggml_cuda_assign_buffers_no_scratch(cache.v_a);
    ggml_cuda_assign_buffers_no_scratch(cache.v_b);
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

static const char* falcon_model_type_name(falcon_e_model type) {
  switch (type) {
    case FALCON_7B:
      return "7B";
    case FALCON_40B:
      return "40B";
    default:
      LLAMA_ASSERT(false);
  }
}

// todo: possibly add that information into ggcc v2
t_finetune_type falcon_detect_finetune(falcon_context* ctx,
                                       std::string model_path) {
  std::string model_lower = model_path;
  std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(),
                 ::tolower);

  for (auto const& x : ctx->vocab.special_tokens) {
    if (x.first == "<|prompter|>") {
      return FINETUNE_OPENASSISTANT;
    }
  }
  if (model_lower.find("wizard") != std::string::npos) {
    return FINETUNE_WIZARD;
  }
  if (model_lower.find("oasst1") != std::string::npos) {
    return FINETUNE_OPENASSIST_V1;
  }
  if (model_lower.find("b-instruct") != std::string::npos) {
    return FINETUNE_FALCONINSTRUCT;
  }
  return FINETUNE_UNSPECIFIED;
}

// dynamically gets all tensors from a layer
std::vector<ggml_tensor*> get_tensors_from_layer(falcon_layer& layer) {
  std::vector<ggml_tensor*> tensors;
  ggml_tensor** tensor_ptr = reinterpret_cast<ggml_tensor**>(
      &layer);  // Cast to the pointer to ggml_tensor pointer

  // Iterate through the members and store their addresses in the vector
  for (std::size_t i = 0; i < sizeof(falcon_layer) / sizeof(ggml_tensor*);
       ++i) {
    tensors.push_back(tensor_ptr[i]);
  }

  return tensors;
}
// get vram size of all tensors in a layer (todo: split handling)
size_t calculate_layer_vram_bytes(const falcon_layer& layer) {
  size_t size = 0;
  auto tensors = get_tensors_from_layer(const_cast<falcon_layer&>(layer));

  // Add the size of each member with GPU backend
  for (const auto& tensor : tensors) {
    if (tensor != nullptr && tensor->backend != GGML_BACKEND_CPU) {
      size += ggml_nbytes(tensor);
    }
  }

  return size;
}

static falcon_model* falcon_model_load_internal(
    const std::string& fname, int n_ctx, int n_batch, int n_gpu_layers,
    int main_gpu, ggml_type memory_type, bool use_mmap, bool use_mlock,
    bool vocab_only, falcon_progress_callback progress_callback,
    void* progress_callback_user_data) {
  falcon_model* model_ = new falcon_model();
  falcon_model& model = *model_;

  std::unique_ptr<falcon_model_loader> ml(
      new falcon_model_loader(fname, use_mmap, vocab_only));

  model.vocab = std::move(ml->file_loaders.at(0)->vocab);
  model.hparams = ml->file_loaders.at(0)->hparams;
  model.n_gpu_layers = n_gpu_layers;

  falcon_file_version file_version = ml->file_loaders.at(0)->file_version;
  auto& hparams = model.hparams;

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

  if (file_version < FALCON_FILE_VERSION_GGJT_V3) {
    if (hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ||
        hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_1 ||
        hparams.ftype == LLAMA_FTYPE_MOSTLY_Q8_0) {
      throw std::runtime_error(
          format("this format is no longer supported (see "
                 "https://github.com/ggerganov/llama.cpp/pull/1508)"));
    }
  }

  if (vocab_only) {
    return model_;
  }

  auto& ctx = model.ctx;
  size_t ctx_size;
  size_t mmapped_size;
  ml->calc_sizes(&ctx_size, &mmapped_size);

  // create the ggml context
  {
    model.buf.resize(ctx_size);
    if (use_mlock) {
      model.mlock_buf.init(model.buf.addr);
      model.mlock_buf.grow_to(model.buf.size);
    }

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/model.buf.size,
        /*.mem_buffer =*/model.buf.addr,
        /*.no_alloc   =*/ml->use_mmap,
    };

    model.ctx = ggml_init(ggml_params);
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

  int64_t vram_reserved = 128 * MB;  // that amount of VRAM is to stay free on
                                     // GPU (needs to become a user parameter)
  size_t vram_overhead = 32 * MB;    // this amount of vram is estimated for non
                                     // weight storage buffers on VRAM
  size_t vram_free = 0;              // for vram simulation below

#if defined(GGML_USE_CUBLAS)
  ggml_cuda_update_gpu_status(-1);
  const GPUStatus* system_gpu_status = ggml_cuda_get_system_gpu_status();
  vram_free = system_gpu_status->total_free_vram;

  if (system_gpu_status->device_vram_reserved[main_gpu] != 0) {
    vram_reserved = system_gpu_status->device_vram_reserved[main_gpu];
  }
  // cublas is used in 16 bit mode, temporary cuda storage/conversion buffers
  // are needed for batch ingestion ( could be run in 16 bit mode without
  // performance downgrade and save half the VRAM)

  if (system_gpu_status->num_devices > 0) {
    if (model.type == FALCON_40B) {
      // if lm_head is not offloaded we'd save vram_overhead += (1016+144)*MB;
      if (model.hparams.ftype != LLAMA_FTYPE_ALL_F32) {
        if (n_batch > 1) {
          vram_overhead += (1016 + 512 + 144 + 50) * MB;
        } else {
          if (hparams.n_vocab % 2) {
            vram_overhead += (1016 + 512 + 144 + 50) * MB;
            fprintf(stderr,
                    "%s: INFO: unoptimized fine-tune weights requires "
                    "additional VRAM per device: %7.2f MB\n",
                    __func__, vram_overhead / MB * 1.0);
          }
        }
      }
    }
    if (model.type == FALCON_7B) {
      // if lm_head is not offloaded we'd save vram_overhead += (563+41)*MB;
      if (model.hparams.ftype != LLAMA_FTYPE_ALL_F32) {
        if (n_batch > 1) {
          vram_overhead += (157 + 563 + 41) * MB;
        } else {
          if (hparams.n_vocab % 2) {
            vram_overhead += (157 + 563 + 41) * MB;
            fprintf(stderr,
                    "%s: INFO: unoptimized fine-tune weights requires "
                    "additional VRAM per device: %7.2f MB\n",
                    __func__, vram_overhead / MB * 1.0);
          }
        }
      }
    }
  } else {
    printf("%s: WARNING: no CUDA devices found, falling back to CPU\n",
           __func__);
  }

#endif

  // prepare memory for the weights
  size_t vram_weights = 0;
  size_t vram_scratch = 0;
  size_t vram_output = 0;  // for display only

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
    // output layer offloading is on by default now, it's one of the biggest CPU
    // consumers
    bool offload_output = true;
    if (n_gpu_layers == 0) offload_output = false;
#ifdef GGML_USE_CUBLAS
    if (system_gpu_status->num_devices == 0) offload_output = false;
#endif

    if (offload_output) {  // NOLINT
      // backend_norm = LLAMA_BACKEND_OFFLOAD; // this requires REPEAT on GPU
      // (in f7b)
      backend_norm = GGML_BACKEND_CPU;
      backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
      if (model.type == FALCON_7B && n_batch > 1) {
        backend_output = LLAMA_BACKEND_OFFLOAD;  // only one n_head_kv
      }
    } else {
      backend_norm = GGML_BACKEND_CPU;
      backend_output = GGML_BACKEND_CPU;
    }
    // backend_output=GGML_BACKEND_CPU;// 1GB vram for 16 bin cublas (with and
    // without n_batch)

    vram_output = 0;
    // "output" tensor
    {
      model.output_norm =
          ml->get_tensor("transformer.ln_f.weight", {n_embd}, backend_norm);
      model.output_norm_b =
          ml->get_tensor("transformer.ln_f.bias", {n_embd}, backend_norm);
      // lm_head does not always run in quantized kernels - cuda currently
      // converts it to 16 bit which will cause 1GB vram overhead on 40B
      model.lm_head =
          ml->get_tensor("lm_head.weight", {n_embd, n_vocab}, backend_output);
    }

    if (backend_norm != GGML_BACKEND_CPU) {
      vram_weights +=
          ggml_nbytes(model.output_norm) + ggml_nbytes(model.output_norm_b);
      vram_output +=
          ggml_nbytes(model.output_norm) + ggml_nbytes(model.output_norm_b);
      vram_free -=
          ggml_nbytes(model.output_norm) + ggml_nbytes(model.output_norm_b);
    }
    if (backend_output != GGML_BACKEND_CPU) {
      vram_weights += ggml_nbytes(model.lm_head);
      vram_output += ggml_nbytes(model.lm_head);
      vram_free -= ggml_nbytes(model.lm_head);
    }

    int i_gpu_start = n_layer - n_gpu_layers;
    if (i_gpu_start < 0)
      i_gpu_start = 0;  // n_gpu_layers can be larger than n_layer

    int i_gpu_last = n_layer;  // allows to terminate the offloading earlier.
                               // TODO: instead do a proper calculation run and
                               // determine the start before the loop

#ifdef GGML_USE_CUBLAS
    if (system_gpu_status->num_devices == 0) {
      i_gpu_start = 999;
      i_gpu_last = -1;
      n_gpu_layers = 0;
      model.n_gpu_layers = 0;
    }
#endif

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

      auto& layer = model.layers[i];

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
#ifdef GGML_USE_CUBLAS
      if (backend != GGML_BACKEND_CPU) {
        size_t vram_layer = 0;
        vram_layer = calculate_layer_vram_bytes(layer) *
                     1.035;  // 3.5%-4.0% too small for some reason
        vram_weights += vram_layer;
        vram_free =
            (vram_layer > vram_free)
                ? 0
                : vram_free -
                      vram_layer;  // simulate the layer being loaded in VRAM
        // test if we have enough VRAM to offload the next layer

        if (i < n_layer &&
            (int64_t)vram_free <= (int64_t)(vram_overhead + vram_scratch +
                                            vram_reserved + vram_layer)) {
          int64_t missing_vram_mb =
              (vram_layer * n_layer + vram_scratch + vram_reserved +
               vram_overhead + vram_output) > system_gpu_status->total_free_vram
                  ? ((vram_layer * n_layer + vram_scratch + vram_reserved +
                      vram_overhead + vram_output) -
                     system_gpu_status->total_free_vram) /
                            MB +
                        1
                  : 0;
          fprintf(stderr,
                  "%s: INFO: not enough VRAM to offload layer %d (missing %zd "
                  "MB)\n",
                  __func__, i + 1, missing_vram_mb);

          model.n_gpu_layers = n_gpu_layers;
          i_gpu_last = i;
          model.i_gpu_last = i_gpu_last;
          n_gpu_layers = i_gpu_last - i_gpu_start;
          fprintf(stderr,
                  "%s: INFO: %d layers will be offloaded to GPU (layers %d to "
                  "%d)\n",
                  __func__, n_gpu_layers, i_gpu_start + 1, i_gpu_last + 1);
        }
      }
#endif
    }
  }

  ml->verify_correct_load();

  // print memory requirements
  {
    int64_t mem_required = ctx_size + mmapped_size -
                           vram_weights +  // weights in VRAM not in memory
                           FALCON_MEM_REQ_SCRATCH0().at(model.type) +
                           FALCON_MEM_REQ_SCRATCH1().at(model.type) +
                           FALCON_MEM_REQ_EVAL_BATCH(model.type, n_batch, n_ctx)
                               .first +  // only prompt context relevant but
                           FALCON_MEM_REQ_EVAL_BATCH(model.type, n_batch, n_ctx)
                               .second +  // only prompt context actually scales
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
  for (falcon_load_tensor& lt : ml->tensors_map.tensors) {
    model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
  }

  if (progress_callback) {
    progress_callback(0.01f, progress_callback_user_data, "Loading weights");
  }

  ml->load_all_data(progress_callback, progress_callback_user_data,
                    use_mlock ? &model.mlock_mmap : NULL);

  if (progress_callback) {
    progress_callback(0.98f, progress_callback_user_data, "Tensors populated");
  }

#if defined(GGML_USE_CUBLAS)
  ggml_cuda_update_gpu_status(-1);
  progress_callback(0.99f, progress_callback_user_data, "Waiting for CUDA");
  // while (!ggml_init_cublas(true))
  //   std::this_thread::sleep_for(std::chrono::milliseconds(50));
  progress_callback(1.0f, progress_callback_user_data,
                    "Tensors populated, CUDA ready");
#else
  progress_callback(1.0f, progress_callback_user_data, "Tensors populated");
#endif

  model.mapping = std::move(ml->mapping);

  // loading time will be recalculate after the first eval, so
  // we take page faults deferred by mmap() into consideration

  return model_;
}

static falcon_model* falcon_model_load(
    const std::string& fname, int n_ctx, int n_batch, int n_gpu_layers,
    int main_gpu, ggml_type memory_type, bool use_mmap, bool use_mlock,
    bool vocab_only, falcon_progress_callback progress_callback,
    void* progress_callback_user_data) {
  try {
    falcon_model* model = falcon_model_load_internal(
        fname, n_ctx, n_batch, n_gpu_layers, main_gpu, memory_type, use_mmap,
        use_mlock, vocab_only, progress_callback, progress_callback_user_data);
    return model;
  } catch (const std::exception& err) {
    fprintf(stderr, "error loading model: %s\n", err.what());
    return nullptr;
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
static bool falcon_eval_internal(falcon_context& lctx,
                                 const falcon_token* tokens, const int n_tokens,
                                 const int n_past, const int n_threads,
                                 const char* cgraph_fname, int debug_timings) {
  const int64_t t_start_us = ggml_time_us();
  bool use_broadcasting = false;  //(n_tokens == 1); // switched from
                                  // interleaving repeat to broadcasting

  const int N = n_tokens;
  // const int N = embd_inp.size();

  const auto& model = lctx.model;
  const auto& hparams = model.hparams;

  const auto& kv_self = model.kv_self;

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

  auto& mem_per_token = lctx.mem_per_token;
  auto& buf_compute = lctx.buf_compute;
  if (n_tokens > 1 && n_gpu_layers > 0) {
    // for batched prompt processing if using cublas on QKV multiplications is
    // wanted this causes a expensive interleaving repeat and cpy on CPU but
    // enabled highspeed processing ín all tested cases CPU processing was
    // faster (through interleaved broadcasting) use_broadcasting=false;
  }

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_compute.size,
      /*.mem_buffer =*/buf_compute.addr,
      /*.no_alloc   =*/false,
  };

  struct ggml_context* ctx0 = ggml_init(params);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are
  // degrading the performance
  ggml_cgraph gf = {};

  struct ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
  ggml_set_name(embd, "embd");
  memcpy(embd->data, tokens, N * ggml_element_size(embd));

  struct ggml_tensor* cur;
  struct ggml_tensor* inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

  struct ggml_tensor* layernorm_output;

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
  // todo: instead of n_layer use either a flag in model/params or a backend
  // test to determine if norm/output are on GPU
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
    if (il >= i_gpu_start && il <= i_gpu_last) {
      offload_func =
          ggml_cuda_assign_buffers;  // sets the output backend to GPU
    }
#endif  // GGML_USE_CUBLAS

    // struct ggml_tensor * inpSA = inpL;

    lctx.use_buf(ctx0, 0);
    // self-attention
    {
      layernorm_output = ggml_norm(ctx0, inpL);

      ggml_tensor* il_a =
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

      struct ggml_tensor* Qcur =
          ggml_view_3d(ctx0, cur, head_dim, n_head, N, head_dim * sizeof_wtype,
                       head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype, 0);
      ggml_set_name(Qcur, "Qcur");

      struct ggml_tensor* Kcur = ggml_view_3d(
          ctx0, cur, head_dim, n_head_kv, N, head_dim * sizeof_wtype,
          head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
          head_dim * n_head * sizeof_wtype);
      ggml_set_name(Kcur, "Kcur");

      struct ggml_tensor* Vcur = ggml_view_3d(
          ctx0, cur, head_dim, n_head_kv, N, head_dim * sizeof_wtype,
          head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
          head_dim * (n_head + n_head_kv) * sizeof_wtype);
      ggml_set_name(Vcur, "Vcur");

      // using mode = 2 for neox mode
      Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, head_dim, 2, n_ctx);
      Kcur = ggml_rope_inplace(ctx0, Kcur, n_past, head_dim, 2, n_ctx);
      // Qcur->meta.f_custom[GGML_CUSTOM_F_ROPE_ANG_SCALE] = 0.25f;
      // Kcur->meta.f_custom[GGML_CUSTOM_F_ROPE_ANG_SCALE] = 0.25f;

      // store key and value to memory
      //{
      struct ggml_tensor* k =
          ggml_view_1d(ctx0, kv_self.k, N * n_head_kv * head_dim,
                       (ggml_element_size(kv_self.k) * n_head_kv * head_dim) *
                           (il * n_ctx + n_past));
      ggml_set_name(k, "k");
      ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
#ifdef FALCON_NO_KV_UPGRADE
      struct ggml_tensor* v =
          ggml_view_1d(ctx0, kv_self.v, N * n_head_kv * head_dim,
                       (ggml_element_size(kv_self.v) * n_head_kv * head_dim) *
                           (il * n_ctx + n_past));
      ggml_set_name(v, "v");
      ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
#endif

#ifndef FALCON_NO_KV_UPGRADE
      struct ggml_tensor* V_prev = ggml_view_3d(
          ctx0, (kv_self.v_current == kv_self.V_A) ? kv_self.v_a : kv_self.v_b,
          n_past, head_dim, n_head_kv,
          /*nb1*/ (n_past)*ggml_element_size(kv_self.v),
          /*nb2*/ head_dim * ggml_element_size(kv_self.v) * (n_past),
          /*off*/ il * n_ctx * ggml_element_size(kv_self.v) * n_head_kv *
              head_dim);
      ggml_set_name(V_prev, "V_prev");
      struct ggml_tensor* V_new = ggml_view_3d(
          ctx0, (kv_self.v_current == kv_self.V_A) ? kv_self.v_b : kv_self.v_a,
          n_past + N, head_dim, n_head_kv,
          /*nb1*/ (N + n_past) * ggml_element_size(kv_self.v),
          /*nb2*/ head_dim * ggml_element_size(kv_self.v) * (N + n_past),
          /*off*/ il * n_ctx * ggml_element_size(kv_self.v) * n_head_kv *
              head_dim);
      ggml_set_name(V_new, "V_new");
      if (n_past == 0) {
        ggml_build_forward_expand(
            &gf, ggml_cpy(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3), V_new));
      } else {
        V_new = ggml_set_inplace(ctx0, V_new, V_prev, V_new->nb[1],
                                 V_new->nb[2], V_new->nb[3], 0);
        V_new = ggml_set_inplace(
            ctx0, V_new, ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3)),
            V_new->nb[1], V_new->nb[2], V_new->nb[3],
            n_past * ggml_element_size(kv_self.v));
        ggml_build_forward_expand(&gf, V_new);
      }
#endif

      //}

      struct ggml_tensor* K = ggml_permute(
          ctx0,
          ggml_view_3d(
              ctx0, kv_self.k, head_dim, n_head_kv, n_past + N,
              head_dim * sizeof_wtype, head_dim * n_head_kv * sizeof_wtype,
              il * n_ctx * ggml_element_size(kv_self.k) * n_head_kv * head_dim),
          0, 2, 1, 3);

      // K * Q
      if (!use_broadcasting) {
        // interleaved repeat for multiplication (now broadcasted)
        struct ggml_tensor* repeat_dummy =
            ggml_new_tensor_3d(ctx0, inpL->type, head_dim, N + n_past, n_head);
        K = ggml_repeat2(ctx0, ggml_cont(ctx0, K), repeat_dummy);
      }

      ggml_set_name(K, "K");
      struct ggml_tensor* Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
      ggml_set_name(Q, "Q");
      struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);
      ggml_set_name(KQ, "KQ");

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      struct ggml_tensor* KQ_scaled = ggml_scale_inplace(
          ctx0, KQ, ggml_new_f32(ctx0, 1.0f / sqrt(float(head_dim))));
      ggml_set_name(KQ_scaled, "KQ_scaled");

      // KQ_masked = mask_past(KQ_scaled)
      struct ggml_tensor* KQ_masked =
          ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
      ggml_set_name(KQ_masked, "KQ_masked");

      // KQ = soft_max(KQ_masked)
      struct ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
      ggml_set_name(KQ_soft_max, "KQ_soft_max");

// V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0,
// 3).contiguous()
#ifdef FALCON_NO_KV_UPGRADE
      struct ggml_tensor* V = ggml_permute(
          ctx0,
          ggml_view_3d(
              ctx0, kv_self.v, head_dim, n_head_kv, n_past + N,
              head_dim * sizeof_wtype, head_dim * n_head_kv * sizeof_wtype,
              il * n_ctx * ggml_element_size(kv_self.v) * n_head_kv * head_dim),
          1, 2, 0, 3);
      V = ggml_cont(ctx0, V);
#else
      struct ggml_tensor* V = V_new;
#endif
      {
        if (!use_broadcasting) {
          // interleaved repeat for multiplication (now broadcasted)
          struct ggml_tensor* repeat_dummy_permuted = ggml_new_tensor_3d(
              ctx0, inpL->type, N + n_past, head_dim, n_head);
          V = ggml_repeat2(ctx0, V, repeat_dummy_permuted);
        }
        ggml_set_name(V, "V");
      }

      // KQV = transpose(V) * KQ_soft_max
      struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
      ggml_set_name(KQV, "KQV");

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
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

    struct ggml_tensor* inpFF = layernorm_output;
    ggml_set_name(inpFF, "inpFF");
    struct ggml_tensor* attn_out =
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
  struct ggml_tensor* embeddings = NULL;

  offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
  if (n_gpu_layers > 0 && n_layer >= i_gpu_start && n_layer <= i_gpu_last) {
    offload_func = ggml_cuda_assign_buffers;  // sets the output backend to GPU
  }
#endif  // GGML_USE_CUBLAS

  // norm
  {
    cur = ggml_norm(ctx0, inpL);
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
#if 0
{
    double used_mem = ggml_used_mem(ctx0) / 1024.0 / 1024.0;
    double scratch_mem_0 = lctx.get_buf_max_mem(0) / 1024.0 / 1024.0;
    double scratch_mem_1 = lctx.get_buf_max_mem(1) / 1024.0 / 1024.0;

    printf("\n%s: Memory Usage\n", __func__);
    printf("tokens: %d - n_batch: %d - n_context: %d\n", n_past+n_tokens, N, n_ctx);
    printf("  Used Memory:    %.3f MB\n", used_mem);
    printf("  Scratch Memory 0: %.3f MB\n", scratch_mem_0);
    printf("  Scratch Memory 1: %.3f MB\n", scratch_mem_1);
}
#endif
  // logits -> probs
  // cur = ggml_soft_max_inplace(ctx0, cur);

  // run the computation
  ggml_build_forward_expand(&gf, cur);

  ggml_backend lm_head_backend = model.lm_head->backend;
  // uneven lm_head from manually added tokens causes cublas errors with 7B
  if (model.type == FALCON_7B && (n_tokens > 1) &&
      model.lm_head->ne[1] % 2 != 0)
    model.lm_head->backend = GGML_BACKEND_CPU;  // cublas fails

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

    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
  }
#else
  ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
#endif
  model.lm_head->backend = lm_head_backend;
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
  // cycle kv/v storage buffer
  lctx.model.kv_self.v_current =
      (kv_self.v_current == kv_self.V_A) ? kv_self.V_B : kv_self.V_A;

  // extract logits
  {
    auto& logits_out = lctx.logits;

    if (lctx.logits_all) {
      logits_out.resize(n_vocab * N);
      memcpy(logits_out.data(), (float*)ggml_get_data(cur),
             sizeof(float) * n_vocab * N);
    } else {
      // return result for just the last token
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(),
             (float*)ggml_get_data(cur) + (n_vocab * (N - 1)),
             sizeof(float) * n_vocab);
    }
  }

  // extract embeddings
  if (!lctx.embedding.empty()) {
    auto& embedding_out = lctx.embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(),
           (float*)ggml_get_data(embeddings) + (n_embd * (N - 1)),
           sizeof(float) * n_embd);
  }

  if (mem_per_token == 0) {
    mem_per_token = ggml_used_mem(ctx0) / N;
  }

#if 0
    double used_mem = ggml_used_mem(ctx0) / 1024.0 / 1024.0;
    double scratch_mem_0 = lctx.get_buf_max_mem(0) / 1024.0 / 1024.0;
    double scratch_mem_1 = lctx.get_buf_max_mem(1) / 1024.0 / 1024.0;

    printf("\n%s: Memory Usage\n", __func__);
    printf("n_past: %d - n_batch: %d - n_context: %d\n", n_past+n_tokens, N, n_ctx);
    printf("  Used Memory:    %.3f MB\n", used_mem);
    printf("  Scratch Memory 0: %.3f MB\n", scratch_mem_0);
    printf("  Scratch Memory 1: %.3f MB\n", scratch_mem_1);
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
// tokenizer - bpe type, gpt2 tokenization compatible
//

struct ggllm_bpe_symbol {
  using index = int;
  index prev;
  index next;
  const char* text;
  size_t n;
};

static_assert(std::is_trivially_copyable<ggllm_bpe_symbol>::value,
              "ggllm_bpe_symbol is not trivially copyable");

struct ggllm_bpe_bigram {
  struct comparator {
    bool operator()(ggllm_bpe_bigram& l, ggllm_bpe_bigram& r) {
      return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
  };

  using queue_storage = std::vector<ggllm_bpe_bigram>;
  using queue =
      std::priority_queue<ggllm_bpe_bigram, queue_storage, comparator>;
  ggllm_bpe_symbol::index left;
  ggllm_bpe_symbol::index right;
  std::string text;
  int rank;
  size_t size;
};

struct falcon_tokenizer {
  falcon_tokenizer(const falcon_vocab& vocab, bool g2ws_) : vocab_(vocab) {
    flag_g2ws = g2ws_;
  }

  void tokenize(const std::string& text,
                std::vector<falcon_vocab::id>& output) {
    int final_prev_index = -1;
    // auto start = ggml_time_us();
    auto word_collection = bpe_gpt2_preprocess(text);
    // auto end = ggml_time_us();
    // fprintf(stderr, "%s: preprocessing took %0.3f ms\n", __func__, (end -
    // start) / 1000.0);

    symbols_final.clear();

    for (auto& word : word_collection) {
      work_queue_ = ggllm_bpe_bigram::queue();
      symbols_.clear();
      bool is_special = false;
      for (auto it = vocab_.special_tokens.begin();
           it != vocab_.special_tokens.end(); ++it) {
        std::string special_token = it->first;
        if (word.compare(special_token) == 0) {
          ggllm_bpe_symbol sym;
          sym.text = word.c_str();
          sym.n = word.size();
          sym.prev = -1;
          sym.next = -1;
          symbols_.emplace_back(sym);
          is_special = true;
          break;
        }
      }

      int index = 0;
      size_t offset = 0;
      if (!is_special) {
        while (offset < word.size()) {
          ggllm_bpe_symbol sym;
          size_t char_len =
              std::min(word.size() - offset,
                       (size_t)CNCTUnicode::utf8_len(word[offset]));
          sym.text = word.c_str() + offset;
          sym.n = 1;
          sym.n = char_len;
          offset += sym.n;
          sym.prev = index - 1;
          sym.next = offset == word.size() ? -1 : index + 1;
          index++;
          symbols_.emplace_back(sym);
        }
        for (size_t i = 1; i < symbols_.size(); ++i) {
          add_new_bigram(i - 1, i);
        }
      }
      // build token(s)
      while (!work_queue_.empty()) {
        auto bigram = work_queue_.top();
        work_queue_.pop();

        auto& left_symbol = symbols_[bigram.left];
        auto& right_symbol = symbols_[bigram.right];

        if (left_symbol.n == 0 || right_symbol.n == 0) {
          continue;
        }
        std::string left_token = std::string(left_symbol.text, left_symbol.n);
        std::string right_token =
            std::string(right_symbol.text, right_symbol.n);
        if (left_token + right_token != bigram.text) {
          continue;  // Skip this bigram if it's outdated
        }

        // merge the right sym into the left one
        left_symbol.n += right_symbol.n;
        right_symbol.n = 0;

        // remove the right sym from the chain
        left_symbol.next = right_symbol.next;
        if (right_symbol.next >= 0) {
          symbols_[right_symbol.next].prev = bigram.left;
        }

        add_new_bigram(left_symbol.prev,
                       bigram.left);  // left side of current symbol
        add_new_bigram(bigram.left,
                       left_symbol.next);  // right side of current symbol
      }
      // add the fnished tokens to the final list keeping correct order for next
      // and prev

      for (auto& sym : symbols_) {
        if (sym.n > 0) {
          sym.prev = final_prev_index;
          sym.next = -1;
          if (final_prev_index != -1) {
            symbols_final[final_prev_index].next = symbols_final.size();
          }
          symbols_final.emplace_back(sym);
          final_prev_index = symbols_final.size() - 1;
        }
      }
    }

    symbols_ = symbols_final;
    if (symbols_.size())
      for (int i = 0; i != -1; i = symbols_[i].next) {
        auto& symbol = symbols_[i];
        if (symbol.n == 0) {
          continue;
        }
        std::string str = std::string(symbol.text, symbol.n);
        std::string str_decoded = decode_token(str);
        auto token = vocab_.token_to_id.find(str_decoded);

        if (token == vocab_.token_to_id.end()) {
          for (auto j = str_decoded.begin(); j != str_decoded.end(); ++j) {
            std::string byte_str(1, *j);
            auto token_multibyte = vocab_.token_to_id.find(byte_str);
            if (token_multibyte == vocab_.token_to_id.end()) {
              fprintf(stderr, "ERROR: byte not found in vocab: '%s'\n",
                      byte_str.c_str());
            }
            output.push_back((*token_multibyte).second);
          }
        } else {
          output.push_back((*token).second);
        }
      }
  }

 private:
  void add_new_bigram(int left, int right) {
    if (left == -1 || right == -1) return;

    std::string left_token = std::string(symbols_[left].text, symbols_[left].n);
    std::string right_token =
        std::string(symbols_[right].text, symbols_[right].n);

    int rank_found = -1;
    rank_found = vocab_.find_bpe_rank(left_token, right_token);

    if (rank_found < 0) {
      return;
    }

    ggllm_bpe_bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.rank = rank_found;
    bigram.size = left_token.size() + right_token.size();
    bigram.text = left_token + right_token;
    work_queue_.push(bigram);
  }

  std::unordered_map<unsigned char, std::string> bytes_to_unicode() {
    static std::unordered_map<unsigned char, std::string> hex_map = {
        {0x21, "\x21"},     {0x22, "\x22"},     {0x23, "\x23"},
        {0x24, "\x24"},     {0x25, "\x25"},     {0x26, "\x26"},
        {0x27, "\x27"},     {0x28, "\x28"},     {0x29, "\x29"},
        {0x2A, "\x2A"},     {0x2B, "\x2B"},     {0x2C, "\x2C"},
        {0x2D, "\x2D"},     {0x2E, "\x2E"},     {0x2F, "\x2F"},
        {0x30, "\x30"},     {0x31, "\x31"},     {0x32, "\x32"},
        {0x33, "\x33"},     {0x34, "\x34"},     {0x35, "\x35"},
        {0x36, "\x36"},     {0x37, "\x37"},     {0x38, "\x38"},
        {0x39, "\x39"},     {0x3A, "\x3A"},     {0x3B, "\x3B"},
        {0x3C, "\x3C"},     {0x3D, "\x3D"},     {0x3E, "\x3E"},
        {0x3F, "\x3F"},     {0x40, "\x40"},     {0x41, "\x41"},
        {0x42, "\x42"},     {0x43, "\x43"},     {0x44, "\x44"},
        {0x45, "\x45"},     {0x46, "\x46"},     {0x47, "\x47"},
        {0x48, "\x48"},     {0x49, "\x49"},     {0x4A, "\x4A"},
        {0x4B, "\x4B"},     {0x4C, "\x4C"},     {0x4D, "\x4D"},
        {0x4E, "\x4E"},     {0x4F, "\x4F"},     {0x50, "\x50"},
        {0x51, "\x51"},     {0x52, "\x52"},     {0x53, "\x53"},
        {0x54, "\x54"},     {0x55, "\x55"},     {0x56, "\x56"},
        {0x57, "\x57"},     {0x58, "\x58"},     {0x59, "\x59"},
        {0x5A, "\x5A"},     {0x5B, "\x5B"},     {0x5C, "\x5C"},
        {0x5D, "\x5D"},     {0x5E, "\x5E"},     {0x5F, "\x5F"},
        {0x60, "\x60"},     {0x61, "\x61"},     {0x62, "\x62"},
        {0x63, "\x63"},     {0x64, "\x64"},     {0x65, "\x65"},
        {0x66, "\x66"},     {0x67, "\x67"},     {0x68, "\x68"},
        {0x69, "\x69"},     {0x6A, "\x6A"},     {0x6B, "\x6B"},
        {0x6C, "\x6C"},     {0x6D, "\x6D"},     {0x6E, "\x6E"},
        {0x6F, "\x6F"},     {0x70, "\x70"},     {0x71, "\x71"},
        {0x72, "\x72"},     {0x73, "\x73"},     {0x74, "\x74"},
        {0x75, "\x75"},     {0x76, "\x76"},     {0x77, "\x77"},
        {0x78, "\x78"},     {0x79, "\x79"},     {0x7A, "\x7A"},
        {0x7B, "\x7B"},     {0x7C, "\x7C"},     {0x7D, "\x7D"},
        {0x7E, "\x7E"},     {0xA1, "\xC2\xA1"}, {0xA2, "\xC2\xA2"},
        {0xA3, "\xC2\xA3"}, {0xA4, "\xC2\xA4"}, {0xA5, "\xC2\xA5"},
        {0xA6, "\xC2\xA6"}, {0xA7, "\xC2\xA7"}, {0xA8, "\xC2\xA8"},
        {0xA9, "\xC2\xA9"}, {0xAA, "\xC2\xAA"}, {0xAB, "\xC2\xAB"},
        {0xAC, "\xC2\xAC"}, {0xAE, "\xC2\xAE"}, {0xAF, "\xC2\xAF"},
        {0xB0, "\xC2\xB0"}, {0xB1, "\xC2\xB1"}, {0xB2, "\xC2\xB2"},
        {0xB3, "\xC2\xB3"}, {0xB4, "\xC2\xB4"}, {0xB5, "\xC2\xB5"},
        {0xB6, "\xC2\xB6"}, {0xB7, "\xC2\xB7"}, {0xB8, "\xC2\xB8"},
        {0xB9, "\xC2\xB9"}, {0xBA, "\xC2\xBA"}, {0xBB, "\xC2\xBB"},
        {0xBC, "\xC2\xBC"}, {0xBD, "\xC2\xBD"}, {0xBE, "\xC2\xBE"},
        {0xBF, "\xC2\xBF"}, {0xC0, "\xC3\x80"}, {0xC1, "\xC3\x81"},
        {0xC2, "\xC3\x82"}, {0xC3, "\xC3\x83"}, {0xC4, "\xC3\x84"},
        {0xC5, "\xC3\x85"}, {0xC6, "\xC3\x86"}, {0xC7, "\xC3\x87"},
        {0xC8, "\xC3\x88"}, {0xC9, "\xC3\x89"}, {0xCA, "\xC3\x8A"},
        {0xCB, "\xC3\x8B"}, {0xCC, "\xC3\x8C"}, {0xCD, "\xC3\x8D"},
        {0xCE, "\xC3\x8E"}, {0xCF, "\xC3\x8F"}, {0xD0, "\xC3\x90"},
        {0xD1, "\xC3\x91"}, {0xD2, "\xC3\x92"}, {0xD3, "\xC3\x93"},
        {0xD4, "\xC3\x94"}, {0xD5, "\xC3\x95"}, {0xD6, "\xC3\x96"},
        {0xD7, "\xC3\x97"}, {0xD8, "\xC3\x98"}, {0xD9, "\xC3\x99"},
        {0xDA, "\xC3\x9A"}, {0xDB, "\xC3\x9B"}, {0xDC, "\xC3\x9C"},
        {0xDD, "\xC3\x9D"}, {0xDE, "\xC3\x9E"}, {0xDF, "\xC3\x9F"},
        {0xE0, "\xC3\xA0"}, {0xE1, "\xC3\xA1"}, {0xE2, "\xC3\xA2"},
        {0xE3, "\xC3\xA3"}, {0xE4, "\xC3\xA4"}, {0xE5, "\xC3\xA5"},
        {0xE6, "\xC3\xA6"}, {0xE7, "\xC3\xA7"}, {0xE8, "\xC3\xA8"},
        {0xE9, "\xC3\xA9"}, {0xEA, "\xC3\xAA"}, {0xEB, "\xC3\xAB"},
        {0xEC, "\xC3\xAC"}, {0xED, "\xC3\xAD"}, {0xEE, "\xC3\xAE"},
        {0xEF, "\xC3\xAF"}, {0xF0, "\xC3\xB0"}, {0xF1, "\xC3\xB1"},
        {0xF2, "\xC3\xB2"}, {0xF3, "\xC3\xB3"}, {0xF4, "\xC3\xB4"},
        {0xF5, "\xC3\xB5"}, {0xF6, "\xC3\xB6"}, {0xF7, "\xC3\xB7"},
        {0xF8, "\xC3\xB8"}, {0xF9, "\xC3\xB9"}, {0xFA, "\xC3\xBA"},
        {0xFB, "\xC3\xBB"}, {0xFC, "\xC3\xBC"}, {0xFD, "\xC3\xBD"},
        {0xFE, "\xC3\xBE"}, {0xFF, "\xC3\xBF"}, {0x00, "\xC4\x80"},
        {0x01, "\xC4\x81"}, {0x02, "\xC4\x82"}, {0x03, "\xC4\x83"},
        {0x04, "\xC4\x84"}, {0x05, "\xC4\x85"}, {0x06, "\xC4\x86"},
        {0x07, "\xC4\x87"}, {0x08, "\xC4\x88"}, {0x09, "\xC4\x89"},
        {0x0A, "\xC4\x8A"}, {0x0B, "\xC4\x8B"}, {0x0C, "\xC4\x8C"},
        {0x0D, "\xC4\x8D"}, {0x0E, "\xC4\x8E"}, {0x0F, "\xC4\x8F"},
        {0x10, "\xC4\x90"}, {0x11, "\xC4\x91"}, {0x12, "\xC4\x92"},
        {0x13, "\xC4\x93"}, {0x14, "\xC4\x94"}, {0x15, "\xC4\x95"},
        {0x16, "\xC4\x96"}, {0x17, "\xC4\x97"}, {0x18, "\xC4\x98"},
        {0x19, "\xC4\x99"}, {0x1A, "\xC4\x9A"}, {0x1B, "\xC4\x9B"},
        {0x1C, "\xC4\x9C"}, {0x1D, "\xC4\x9D"}, {0x1E, "\xC4\x9E"},
        {0x1F, "\xC4\x9F"}, {0x20, "\xC4\xA0"}, {0x7F, "\xC4\xA1"},
        {0x80, "\xC4\xA2"}, {0x81, "\xC4\xA3"}, {0x82, "\xC4\xA4"},
        {0x83, "\xC4\xA5"}, {0x84, "\xC4\xA6"}, {0x85, "\xC4\xA7"},
        {0x86, "\xC4\xA8"}, {0x87, "\xC4\xA9"}, {0x88, "\xC4\xAA"},
        {0x89, "\xC4\xAB"}, {0x8A, "\xC4\xAC"}, {0x8B, "\xC4\xAD"},
        {0x8C, "\xC4\xAE"}, {0x8D, "\xC4\xAF"}, {0x8E, "\xC4\xB0"},
        {0x8F, "\xC4\xB1"}, {0x90, "\xC4\xB2"}, {0x91, "\xC4\xB3"},
        {0x92, "\xC4\xB4"}, {0x93, "\xC4\xB5"}, {0x94, "\xC4\xB6"},
        {0x95, "\xC4\xB7"}, {0x96, "\xC4\xB8"}, {0x97, "\xC4\xB9"},
        {0x98, "\xC4\xBA"}, {0x99, "\xC4\xBB"}, {0x9A, "\xC4\xBC"},
        {0x9B, "\xC4\xBD"}, {0x9C, "\xC4\xBE"}, {0x9D, "\xC4\xBF"},
        {0x9E, "\xC5\x80"}, {0x9F, "\xC5\x81"}, {0xA0, "\xC5\x82"},
        {0xAD, "\xC5\x83"}};
    return hex_map;
  }

  std::unordered_map<std::string, unsigned char> unicode_to_bytes() {
    static std::unordered_map<std::string, unsigned char> hex_map = {
        {"\x21", 0x21},     {"\x22", 0x22},     {"\x23", 0x23},
        {"\x24", 0x24},     {"\x25", 0x25},     {"\x26", 0x26},
        {"\x27", 0x27},     {"\x28", 0x28},     {"\x29", 0x29},
        {"\x2A", 0x2A},     {"\x2B", 0x2B},     {"\x2C", 0x2C},
        {"\x2D", 0x2D},     {"\x2E", 0x2E},     {"\x2F", 0x2F},
        {"\x30", 0x30},     {"\x31", 0x31},     {"\x32", 0x32},
        {"\x33", 0x33},     {"\x34", 0x34},     {"\x35", 0x35},
        {"\x36", 0x36},     {"\x37", 0x37},     {"\x38", 0x38},
        {"\x39", 0x39},     {"\x3A", 0x3A},     {"\x3B", 0x3B},
        {"\x3C", 0x3C},     {"\x3D", 0x3D},     {"\x3E", 0x3E},
        {"\x3F", 0x3F},     {"\x40", 0x40},     {"\x41", 0x41},
        {"\x42", 0x42},     {"\x43", 0x43},     {"\x44", 0x44},
        {"\x45", 0x45},     {"\x46", 0x46},     {"\x47", 0x47},
        {"\x48", 0x48},     {"\x49", 0x49},     {"\x4A", 0x4A},
        {"\x4B", 0x4B},     {"\x4C", 0x4C},     {"\x4D", 0x4D},
        {"\x4E", 0x4E},     {"\x4F", 0x4F},     {"\x50", 0x50},
        {"\x51", 0x51},     {"\x52", 0x52},     {"\x53", 0x53},
        {"\x54", 0x54},     {"\x55", 0x55},     {"\x56", 0x56},
        {"\x57", 0x57},     {"\x58", 0x58},     {"\x59", 0x59},
        {"\x5A", 0x5A},     {"\x5B", 0x5B},     {"\x5C", 0x5C},
        {"\x5D", 0x5D},     {"\x5E", 0x5E},     {"\x5F", 0x5F},
        {"\x60", 0x60},     {"\x61", 0x61},     {"\x62", 0x62},
        {"\x63", 0x63},     {"\x64", 0x64},     {"\x65", 0x65},
        {"\x66", 0x66},     {"\x67", 0x67},     {"\x68", 0x68},
        {"\x69", 0x69},     {"\x6A", 0x6A},     {"\x6B", 0x6B},
        {"\x6C", 0x6C},     {"\x6D", 0x6D},     {"\x6E", 0x6E},
        {"\x6F", 0x6F},     {"\x70", 0x70},     {"\x71", 0x71},
        {"\x72", 0x72},     {"\x73", 0x73},     {"\x74", 0x74},
        {"\x75", 0x75},     {"\x76", 0x76},     {"\x77", 0x77},
        {"\x78", 0x78},     {"\x79", 0x79},     {"\x7A", 0x7A},
        {"\x7B", 0x7B},     {"\x7C", 0x7C},     {"\x7D", 0x7D},
        {"\x7E", 0x7E},     {"\xC2\xA1", 0xA1}, {"\xC2\xA2", 0xA2},
        {"\xC2\xA3", 0xA3}, {"\xC2\xA4", 0xA4}, {"\xC2\xA5", 0xA5},
        {"\xC2\xA6", 0xA6}, {"\xC2\xA7", 0xA7}, {"\xC2\xA8", 0xA8},
        {"\xC2\xA9", 0xA9}, {"\xC2\xAA", 0xAA}, {"\xC2\xAB", 0xAB},
        {"\xC2\xAC", 0xAC}, {"\xC2\xAE", 0xAE}, {"\xC2\xAF", 0xAF},
        {"\xC2\xB0", 0xB0}, {"\xC2\xB1", 0xB1}, {"\xC2\xB2", 0xB2},
        {"\xC2\xB3", 0xB3}, {"\xC2\xB4", 0xB4}, {"\xC2\xB5", 0xB5},
        {"\xC2\xB6", 0xB6}, {"\xC2\xB7", 0xB7}, {"\xC2\xB8", 0xB8},
        {"\xC2\xB9", 0xB9}, {"\xC2\xBA", 0xBA}, {"\xC2\xBB", 0xBB},
        {"\xC2\xBC", 0xBC}, {"\xC2\xBD", 0xBD}, {"\xC2\xBE", 0xBE},
        {"\xC2\xBF", 0xBF}, {"\xC3\x80", 0xC0}, {"\xC3\x81", 0xC1},
        {"\xC3\x82", 0xC2}, {"\xC3\x83", 0xC3}, {"\xC3\x84", 0xC4},
        {"\xC3\x85", 0xC5}, {"\xC3\x86", 0xC6}, {"\xC3\x87", 0xC7},
        {"\xC3\x88", 0xC8}, {"\xC3\x89", 0xC9}, {"\xC3\x8A", 0xCA},
        {"\xC3\x8B", 0xCB}, {"\xC3\x8C", 0xCC}, {"\xC3\x8D", 0xCD},
        {"\xC3\x8E", 0xCE}, {"\xC3\x8F", 0xCF}, {"\xC3\x90", 0xD0},
        {"\xC3\x91", 0xD1}, {"\xC3\x92", 0xD2}, {"\xC3\x93", 0xD3},
        {"\xC3\x94", 0xD4}, {"\xC3\x95", 0xD5}, {"\xC3\x96", 0xD6},
        {"\xC3\x97", 0xD7}, {"\xC3\x98", 0xD8}, {"\xC3\x99", 0xD9},
        {"\xC3\x9A", 0xDA}, {"\xC3\x9B", 0xDB}, {"\xC3\x9C", 0xDC},
        {"\xC3\x9D", 0xDD}, {"\xC3\x9E", 0xDE}, {"\xC3\x9F", 0xDF},
        {"\xC3\xA0", 0xE0}, {"\xC3\xA1", 0xE1}, {"\xC3\xA2", 0xE2},
        {"\xC3\xA3", 0xE3}, {"\xC3\xA4", 0xE4}, {"\xC3\xA5", 0xE5},
        {"\xC3\xA6", 0xE6}, {"\xC3\xA7", 0xE7}, {"\xC3\xA8", 0xE8},
        {"\xC3\xA9", 0xE9}, {"\xC3\xAA", 0xEA}, {"\xC3\xAB", 0xEB},
        {"\xC3\xAC", 0xEC}, {"\xC3\xAD", 0xED}, {"\xC3\xAE", 0xEE},
        {"\xC3\xAF", 0xEF}, {"\xC3\xB0", 0xF0}, {"\xC3\xB1", 0xF1},
        {"\xC3\xB2", 0xF2}, {"\xC3\xB3", 0xF3}, {"\xC3\xB4", 0xF4},
        {"\xC3\xB5", 0xF5}, {"\xC3\xB6", 0xF6}, {"\xC3\xB7", 0xF7},
        {"\xC3\xB8", 0xF8}, {"\xC3\xB9", 0xF9}, {"\xC3\xBA", 0xFA},
        {"\xC3\xBB", 0xFB}, {"\xC3\xBC", 0xFC}, {"\xC3\xBD", 0xFD},
        {"\xC3\xBE", 0xFE}, {"\xC3\xBF", 0xFF}, {"\xC4\x80", 0x00},
        {"\xC4\x81", 0x01}, {"\xC4\x82", 0x02}, {"\xC4\x83", 0x03},
        {"\xC4\x84", 0x04}, {"\xC4\x85", 0x05}, {"\xC4\x86", 0x06},
        {"\xC4\x87", 0x07}, {"\xC4\x88", 0x08}, {"\xC4\x89", 0x09},
        {"\xC4\x8A", 0x0A}, {"\xC4\x8B", 0x0B}, {"\xC4\x8C", 0x0C},
        {"\xC4\x8D", 0x0D}, {"\xC4\x8E", 0x0E}, {"\xC4\x8F", 0x0F},
        {"\xC4\x90", 0x10}, {"\xC4\x91", 0x11}, {"\xC4\x92", 0x12},
        {"\xC4\x93", 0x13}, {"\xC4\x94", 0x14}, {"\xC4\x95", 0x15},
        {"\xC4\x96", 0x16}, {"\xC4\x97", 0x17}, {"\xC4\x98", 0x18},
        {"\xC4\x99", 0x19}, {"\xC4\x9A", 0x1A}, {"\xC4\x9B", 0x1B},
        {"\xC4\x9C", 0x1C}, {"\xC4\x9D", 0x1D}, {"\xC4\x9E", 0x1E},
        {"\xC4\x9F", 0x1F}, {"\xC4\xA0", 0x20}, {"\xC4\xA1", 0x7F},
        {"\xC4\xA2", 0x80}, {"\xC4\xA3", 0x81}, {"\xC4\xA4", 0x82},
        {"\xC4\xA5", 0x83}, {"\xC4\xA6", 0x84}, {"\xC4\xA7", 0x85},
        {"\xC4\xA8", 0x86}, {"\xC4\xA9", 0x87}, {"\xC4\xAA", 0x88},
        {"\xC4\xAB", 0x89}, {"\xC4\xAC", 0x8A}, {"\xC4\xAD", 0x8B},
        {"\xC4\xAE", 0x8C}, {"\xC4\xAF", 0x8D}, {"\xC4\xB0", 0x8E},
        {"\xC4\xB1", 0x8F}, {"\xC4\xB2", 0x90}, {"\xC4\xB3", 0x91},
        {"\xC4\xB4", 0x92}, {"\xC4\xB5", 0x93}, {"\xC4\xB6", 0x94},
        {"\xC4\xB7", 0x95}, {"\xC4\xB8", 0x96}, {"\xC4\xB9", 0x97},
        {"\xC4\xBA", 0x98}, {"\xC4\xBB", 0x99}, {"\xC4\xBC", 0x9A},
        {"\xC4\xBD", 0x9B}, {"\xC4\xBE", 0x9C}, {"\xC4\xBF", 0x9D},
        {"\xC5\x80", 0x9E}, {"\xC5\x81", 0x9F}, {"\xC5\x82", 0xA0},
        {"\xC5\x83", 0xAD}};
    return hex_map;
  }
  // len must be available
  bool inline str_is_equal(const char* str1, const char* str2, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      if (str1[i] != str2[i]) {
        return false;
      }
    }
    return true;
  }
  std::vector<std::string> bpe_gpt2_preprocess(const std::string& text) {
    static std::unordered_map<unsigned char, std::string> byte_encoder =
        bytes_to_unicode();
    std::vector<std::string> bpe_words;
    std::vector<std::string> bpe_encoded_words;

    std::string token = "";
    const char* raw_text_p = text.c_str();
    // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
    // ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    bool collecting_numeric = false;
    bool collecting_letter = false;
    bool collecting_special = false;
    bool collecting_whitespace_lookahead = false;
    bool collecting = false;

    std::vector<CNCTString> text_utf;
    text_utf.reserve(text.size());
    bpe_words.reserve(text.size());
    bpe_encoded_words.reserve(text.size());

    text_utf = CNCTUnicode::split_utf8_enhanced(text);
    std::map<std::string, int> special_tokens = vocab_.special_tokens;
    int smallest_len_special_tokens = 0;
    if (special_tokens.size()) {
      smallest_len_special_tokens = special_tokens.begin()->first.size();
      for (auto it = special_tokens.begin(); it != special_tokens.end(); ++it) {
        if (it->first.size() < (size_t)smallest_len_special_tokens)
          smallest_len_special_tokens = it->first.size();
      }
    }

    for (int i = 0; i < (int)text_utf.size(); i++) {
      const CNCTString& utf_char = text_utf[i];
      bool split_condition = false;
      const char* text_pos = raw_text_p + utf_char.seq_offset_bytes;
      int bytes_remain = strlen(text_pos);
      // forward backward lookups
      const CNCTString& utf_char_next =
          (i + 1 < (int)text_utf.size()) ? text_utf[i + 1] : CNCTString();
      const CNCTString& utf_char_next_next =
          (i + 2 < (int)text_utf.size()) ? text_utf[i + 2] : CNCTString();
      // const CNCTString &utf_char_prev = (i > 0) ? text_utf[i-1] :
      // CNCTString();

      // handling special tokens
      bool special_token_found = false;
      if (bytes_remain >= (int)smallest_len_special_tokens)
        for (auto it = special_tokens.begin(); it != special_tokens.end();
             ++it) {
          if ((bytes_remain) < (int)it->first.size()) continue;

          if (str_is_equal(text_pos, it->first.c_str(), it->first.size())) {
            if (token.size()) {
              bpe_words.emplace_back(token);  // push previous content as token
              token.clear();
              collecting = false;
              collecting_letter = false;
              collecting_numeric = false;
              collecting_special = false;
              collecting_whitespace_lookahead = false;
            }

            bpe_words.emplace_back(it->first);  // push special token as token

            // we now advance i until the token is fulfilled by the utf_chars
            int st_bytes = (int)it->first.size();
            for (; st_bytes; st_bytes -= text_utf[i++].str.size())
              ;
            i--;
            special_token_found = true;
            break;
          }
        }

      if (special_token_found) continue;

      // handling contractions
      if (!split_condition && bytes_remain >= 2) {
        // 's|'t|'m|'d
        if (utf_char == '\'' && (utf_char_next == 's' || utf_char_next == 't' ||
                                 utf_char_next == 'm' || utf_char_next == 'd'))
          split_condition = true;
        if (split_condition) {
          if (token.size())
            bpe_words.emplace_back(token);  // push previous content as token
          token = utf_char.str + utf_char_next.str;
          bpe_words.emplace_back(token);
          token = "";
          i++;
          continue;
        }
      }
      if (!split_condition && bytes_remain >= 3) {
        // 're|'ve|'ll
        if (utf_char == '\'' &&
            ((utf_char_next == 'r' || utf_char_next_next == 'e') ||
             (utf_char_next == 'v' || utf_char_next_next == 'e') ||
             (utf_char_next == 'l' || utf_char_next_next == 'l')))
          split_condition = true;
        if (split_condition) {
          // current token + next token can be defined
          if (token.size())
            bpe_words.emplace_back(token);  // push previous content as token
          token = utf_char.str + utf_char_next.str + utf_char_next_next.str;
          bpe_words.emplace_back(token);  // the contraction
          token = "";
          i += 2;
          continue;
        }
      }

      if (!split_condition && !collecting) {
        if (utf_char.char_type == CNCTCharType::LETTER ||
            (!token.size() && utf_char == " " &&
             utf_char_next.char_type == CNCTCharType::LETTER)) {
          collecting_letter = true;
          collecting = true;
        } else if (utf_char.char_type == CNCTCharType::DIGIT ||
                   (!token.size() && utf_char == " " &&
                    utf_char_next.char_type == CNCTCharType::DIGIT)) {
          collecting_numeric = true;
          collecting = true;
        } else if (((utf_char.char_type != CNCTCharType::LETTER &&
                     utf_char.char_type != CNCTCharType::DIGIT) &&
                    (utf_char.char_type != CNCTCharType::WHITESPACE)) ||
                   (!token.size() && utf_char == " " &&
                    utf_char_next.char_type != CNCTCharType::LETTER &&
                    utf_char_next.char_type != CNCTCharType::DIGIT &&
                    utf_char_next.char_type != CNCTCharType::WHITESPACE)) {
          collecting_special = true;
          collecting = true;
        } else if (utf_char.char_type == CNCTCharType::WHITESPACE &&
                   utf_char_next.char_type == CNCTCharType::WHITESPACE) {
          collecting_whitespace_lookahead = true;
          collecting = true;
        } else if (utf_char.char_type == CNCTCharType::WHITESPACE) {
          split_condition = true;
        }
      } else if (!split_condition && collecting) {
        if (collecting_letter && utf_char.char_type != CNCTCharType::LETTER) {
          split_condition = true;
        } else if (collecting_numeric &&
                   utf_char.char_type != CNCTCharType::DIGIT) {
          split_condition = true;
        } else if (collecting_special &&
                   (utf_char.char_type == CNCTCharType::LETTER ||
                    utf_char.char_type == CNCTCharType::DIGIT ||
                    utf_char.char_type == CNCTCharType::WHITESPACE)) {
          split_condition = true;
        } else if (collecting_whitespace_lookahead &&
                   utf_char_next.char_type != CNCTCharType::WHITESPACE) {
          split_condition = true;
        }
      }
      if (utf_char_next.str.size() == 0) {
        split_condition = true;  // final
        token += utf_char.str;
      }

      if (split_condition) {
        if (token.size()) bpe_words.emplace_back(token);
        token = utf_char.str;
        collecting = false;
        collecting_letter = false;
        collecting_numeric = false;
        collecting_special = false;
        collecting_whitespace_lookahead = false;
      } else
        token += utf_char.str;
    }

    for (std::string& word : bpe_words) {
      std::string encoded_token = "";
      for (char& c : word) {
        encoded_token += byte_encoder[c];
      }
      bpe_encoded_words.emplace_back(encoded_token);
    }

    return bpe_encoded_words;
  }

  // decoder (for one token)
  std::string decode_token(const std::string& token) {
    static std::unordered_map<std::string, unsigned char> byte_decoder =
        unicode_to_bytes();
    std::string decoded_token = "";
    auto unicode_seqeunces = CNCTUnicode::split_utf8(token);
    for (auto& unicode_sequence : unicode_seqeunces) {
      decoded_token += byte_decoder[unicode_sequence];
    }

    return decoded_token;
  }

  const falcon_vocab& vocab_;
  std::vector<ggllm_bpe_symbol> symbols_;
  std::vector<ggllm_bpe_symbol> symbols_final;
  ggllm_bpe_bigram::queue work_queue_;
  bool flag_g2ws = false;
};

static std::vector<falcon_vocab::id> falcon_tokenize(const falcon_vocab& vocab,
                                                     const std::string& text,
                                                     bool bos, bool g2ws) {
  falcon_tokenizer tokenizer(vocab, g2ws);
  std::vector<falcon_vocab::id> output;

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
// softmax normalize
void falcon_sample_softmax(struct falcon_context* ctx,
                           falcon_token_data_array* candidates) {
  assert(candidates->size > 0);

  const int64_t t_start_sample_us = ggml_time_us();

  // Sort the logits in descending order
  if (!candidates->sorted) {
    std::sort(candidates->data, candidates->data + candidates->size,
              [](const falcon_token_data& a, const falcon_token_data& b) {
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

void falcon_sample_log_softmax(struct falcon_context* ctx,
                               falcon_token_data_array* candidates) {
  assert(candidates->size > 0);

  const int64_t t_start_sample_us = ggml_time_us();

  // Sort the logits in descending order
  if (!candidates->sorted) {
    std::sort(candidates->data, candidates->data + candidates->size,
              [](const falcon_token_data& a, const falcon_token_data& b) {
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
    candidates->data[i].p = logf(candidates->data[i].p / cum_sum);
  }

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}
// top_k - cut pool to k best candidates (defaults ~ 40)
void falcon_sample_top_k(struct falcon_context* ctx,
                         falcon_token_data_array* candidates, int k,
                         size_t min_keep) {
  const int64_t t_start_sample_us = ggml_time_us();

  k = std::max(k, (int)min_keep);
  k = std::min(k, (int)candidates->size);

  // Sort scores in descending order
  if (!candidates->sorted) {
    auto comp = [](const falcon_token_data& a, const falcon_token_data& b) {
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

// top_p - cut pool to tokens with cumulative probability > p (defaults 1.0)
void falcon_sample_top_p(struct falcon_context* ctx,
                         falcon_token_data_array* candidates, float p,
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

// remove low probability tail (not too useful with low top_k)
void falcon_sample_tail_free(struct falcon_context* ctx,
                             falcon_token_data_array* candidates, float z,
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
  for (float& value : second_derivatives) {
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

// favor more typical tokens out of the pool based on the randomness of the
// total pool distribution
void falcon_sample_typical(struct falcon_context* ctx,
                           falcon_token_data_array* candidates, float p,
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
  std::vector<falcon_token_data> new_candidates;
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
// increases the absolute candidate values, another softmax will make it more
// peaky at low temperatures (high scale)
void falcon_sample_temperature(struct falcon_context* ctx,
                               falcon_token_data_array* candidates_p,
                               float temp) {
  const int64_t t_start_sample_us = ggml_time_us();

  for (size_t i = 0; i < candidates_p->size; ++i) {
    candidates_p->data[i].logit /= temp;
  }

  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  }
}

void falcon_sample_repetition_penalty(struct falcon_context* ctx,
                                      falcon_token_data_array* candidates,
                                      const falcon_token* last_tokens,
                                      size_t last_tokens_size, float penalty) {
  if (last_tokens_size == 0 || penalty == 1.0f) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  for (size_t i = 0; i < candidates->size; ++i) {
    const auto* token_iter = std::find(
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
    struct falcon_context* ctx, falcon_token_data_array* candidates,
    const falcon_token* last_tokens_p, size_t last_tokens_size,
    float alpha_frequency, float alpha_presence) {
  if (last_tokens_size == 0 ||
      (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
    return;
  }

  const int64_t t_start_sample_us = ggml_time_us();

  // Create a frequency map to count occurrences of each token in last_tokens
  std::unordered_map<falcon_token, int> token_count;
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

falcon_token falcon_sample_token_mirostat(struct falcon_context* ctx,
                                          falcon_token_data_array* candidates,
                                          float tau, float eta, int m,
                                          float* mu) {
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
  falcon_token X = falcon_sample_token(ctx, candidates);
  t_start_sample_us = ggml_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const falcon_token_data& candidate) {
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

falcon_token falcon_sample_token_mirostat_v2(
    struct falcon_context* ctx, falcon_token_data_array* candidates, float tau,
    float eta, float* mu) {
  assert(ctx);
  int64_t t_start_sample_us;
  t_start_sample_us = ggml_time_us();

  falcon_sample_softmax(ctx, candidates);

  // Truncate the words with surprise values greater than mu
  candidates->size = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const falcon_token_data& candidate) {
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
  falcon_token X = falcon_sample_token(ctx, candidates);
  t_start_sample_us = ggml_time_us();

  // Compute error as the difference between observed surprise and target
  // surprise value
  size_t X_idx = std::distance(
      candidates->data,
      std::find_if(candidates->data, candidates->data + candidates->size,
                   [&](const falcon_token_data& candidate) {
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

falcon_token falcon_sample_token_greedy(struct falcon_context* ctx,
                                        falcon_token_data_array* candidates) {
  const int64_t t_start_sample_us = ggml_time_us();

  // Find max element
  auto* max_iter = std::max_element(
      candidates->data, candidates->data + candidates->size,
      [](const falcon_token_data& a, const falcon_token_data& b) {
        return a.logit < b.logit;
      });

  falcon_token result = max_iter->id;
  if (ctx) {
    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
  }
  return result;
}

falcon_token falcon_sample_token(struct falcon_context* ctx,
                                 falcon_token_data_array* candidates) {
  assert(ctx);
  const int64_t t_start_sample_us = ggml_time_us();
  falcon_sample_softmax(nullptr, candidates);

  std::vector<float> probs;
  probs.reserve(candidates->size);
  for (size_t i = 0; i < candidates->size; ++i) {
    probs.push_back(candidates->data[i].p);
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  auto& rng = ctx->rng;
  int idx = dist(rng);

  falcon_token result = candidates->data[idx].id;

  ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
  ctx->n_sample++;
  return result;
}

// set context memory buffers
void falcon_context_set_buffers(falcon_context* ctx, int n_batch, int n_ctx) {
  LLAMA_ASSERT(ctx->model.type != FALCON_UNKNOWN);
  ctx->buf_compute.resize(FALCON_MEM_REQ_EVAL().at(ctx->model.type));
  ctx->buf_scratch[0].resize(
      FALCON_MEM_REQ_SCRATCH0().at(ctx->model.type) +
      FALCON_MEM_REQ_EVAL_BATCH(ctx->model.type, n_batch, n_ctx).first);
  ctx->buf_scratch[1].resize(
      FALCON_MEM_REQ_SCRATCH1().at(ctx->model.type) +
      FALCON_MEM_REQ_EVAL_BATCH(ctx->model.type, n_batch, n_ctx).second);
  // fprintf(stderr, "Buffers: compute %.2f MB, scratch0 %.2f MB, scratch1 %.2f
  // MB\n", FALCON_MEM_REQ_EVAL().at(ctx->model.type)/1024.0/1024.0,
  // (FALCON_MEM_REQ_SCRATCH0().at(ctx->model.type)+FALCON_MEM_REQ_EVAL_BATCH(ctx->model.type,n_batch,n_ctx).first)/1024.0/1024.0,
  // (FALCON_MEM_REQ_SCRATCH1().at(ctx->model.type)+FALCON_MEM_REQ_EVAL_BATCH(ctx->model.type,n_batch,n_ctx).second)/1024.0/1024.0);
}
// create a new context with KV cache - if model type is set
// falcon_context_set_buffers() is called as well
struct falcon_context* falcon_context_prepare(falcon_context_params params,
                                              falcon_model* model,
                                              std::string context_name,
                                              bool verbose) {
  falcon_context* ctx = new falcon_context(
      *model, model->vocab);  // ctx model/vocab only references to the globals
  ctx->context_name = context_name;
  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  ctx->rng = std::mt19937(params.seed);
  ctx->logits_all = params.logits_all;

  ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;
  ctx->t_start_us = ggml_time_us();

  if (!params.vocab_only) {
    // reserve memory for context buffers
    if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type,
                       ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
      fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n",
              __func__);
      falcon_free(ctx);
      return nullptr;
    }

    const auto& hparams = ctx->model.hparams;

    // resized during inference
    if (params.logits_all) {
      ctx->logits.reserve(hparams.n_ctx * hparams.n_vocab);
    } else {
      ctx->logits.reserve(hparams.n_vocab);
    }

    if (params.embedding) {
      ctx->embedding.resize(hparams.n_embd);
    }

    if (verbose && model->type != FALCON_UNKNOWN) {
      const size_t memory_size =
          ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
      falcon_context_set_buffers(ctx, params.n_batch, params.n_ctx);
    }
  }
#ifdef GGML_USE_METAL
  if (params.n_gpu_layers > 0) {
    // this allocates all Metal resources and memory buffers
    ctx->ctx_metal = ggml_metal_init(1);

    void* data_ptr = NULL;
    size_t data_size = 0;
    if (params.use_mmap) {
      data_ptr = ctx->model.mapping->addr;
      data_size = ctx->model.mapping->size;
    } else {
      data_ptr = ggml_get_mem_buffer(ctx->model.ctx);
      data_size = ggml_get_mem_size(ctx->model.ctx);
    }

    const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

#define LLAMA_METAL_CHECK_BUF(result)                        \
  if (!(result)) {                                           \
    fprintf(stderr, "%s: failed to add buffer\n", __func__); \
    falcon_free(ctx);                                        \
    return NULL;                                             \
  }

    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "data",
                                                data_ptr, data_size, max_size));

    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "eval",
                                                ctx->buf_compute.addr,
                                                ctx->buf_compute.size, 0));
    LLAMA_METAL_CHECK_BUF(
        ggml_metal_add_buffer(ctx->ctx_metal, "kv", ctx->model.kv_self.buf.addr,
                              ctx->model.kv_self.buf.size, 0));

    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr0",
                                                ctx->buf_scratch[0].addr,
                                                ctx->buf_scratch[0].size, 0));
    LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr1",
                                                ctx->buf_scratch[1].addr,
                                                ctx->buf_scratch[1].size, 0));
#undef LLAMA_METAL_CHECK_BUF
  }
#endif

  return ctx;
}

struct falcon_model* falcon_get_falcon_model(falcon_context* ctx) {
  return &ctx->model;
}

struct falcon_context* falcon_init_from_file(
    const char* path_model, struct falcon_context_params params) {
  ggml_time_init();

  unsigned cur_percentage = 0;
  if (params.progress_callback == NULL) {
    params.progress_callback_user_data =
        &cur_percentage;  // not sure why this is so complicated ? I left it for
                          // now
    params.progress_callback = [](float progress, void* ctx,
                                  const char* status) {
      unsigned percentage = (unsigned)(100 * progress);
      unsigned* cur_percentage_p = (unsigned*)ctx;
      static const int bar_width = 50;
      bool completed = false;
      if (percentage >= 100) {
        completed = true;
        if (!strlen(status)) status = "Completed";
      }

      if (percentage > *cur_percentage_p) {
        *cur_percentage_p = percentage;
      }
    };
  }
  int64_t t_start_us = ggml_time_us();
  ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;
  falcon_model* model = falcon_model_load(
      path_model, params.n_ctx, params.n_batch, params.n_gpu_layers,
      params.main_gpu, memory_type, params.use_mmap, params.use_mlock,
      params.vocab_only, params.progress_callback,
      params.progress_callback_user_data);
  if (model == nullptr) {
    fprintf(stderr, "%s: failed to load model\n", __func__);
    // falcon_free(f_ctx);
    return nullptr;
  }
  // model_load_internal() may change this if VRAM runs out
  params.n_gpu_layers = model->n_gpu_layers;
  params.i_gpu_start =
      model->i_gpu_start;                 // first layer that's GPU accelerated
  params.i_gpu_last = model->i_gpu_last;  // last layer that's GPU accelerated
  falcon_context* f_ctx =
      falcon_context_prepare(params, model, "falcon_main", true);
  f_ctx->t_load_us = ggml_time_us() - t_start_us;
  f_ctx->t_start_us = t_start_us;
  // falcon_context_set_buffers(f_ctx,params.n_batch,params.n_ctx);
  // const size_t memory_size = ggml_nbytes(model->kv_self.k) +
  // ggml_nbytes(model->kv_self.v); fprintf(stderr, "%s: RAM buffers - key_val =
  // %7.2f MB, Compute = %7.2f MB, Scratch 0 = %7.2f MB, Scratch 1 = %7.2f MB
  // \n", __func__, memory_size / 1024.0 / 1024.0, f_ctx->buf_compute.size
  // /1024.0/1024.0, (f_ctx->buf_scratch[0].size)/1024.0/1024.0,
  // (f_ctx->buf_scratch[1].size)/1024.0/1024.0);

  return f_ctx;
}

void falcon_free(struct falcon_context* f_ctx) { delete f_ctx; }

int falcon_apply_lora_from_file_internal(struct falcon_context* ctx,
                                         const char* path_lora,
                                         const char* path_base_model,
                                         int n_threads) {
  fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n",
          __func__, path_lora);

  auto& model = ctx->model;

  const int64_t t_start_lora_us = ggml_time_us();

  auto fin = std::ifstream(path_lora, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
    return 1;
  }

  // verify magic and version
  {
    uint32_t magic;
    fin.read((char*)&magic, sizeof(magic));
    if (magic != LLAMA_FILE_MAGIC_GGLA) {
      fprintf(stderr, "%s: bad file magic\n", __func__);
      return 1;
    }
    uint32_t format_version;
    fin.read((char*)&format_version, sizeof(format_version));

    if (format_version != 1) {
      fprintf(stderr, "%s: unsupported file version\n", __func__);
      return 1;
    }
  }

  int32_t lora_r;
  int32_t lora_alpha;
  fin.read((char*)&lora_r, sizeof(lora_r));
  fin.read((char*)&lora_alpha, sizeof(lora_alpha));
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

  ggml_context* lora_ctx = ggml_init(params);
  std::unordered_map<std::string, struct ggml_tensor*> lora_tensors;

  // create a name -> tensor map of the model to accelerate lookups
  std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
  for (auto& kv : model.tensors_by_name) {
    model_tensors.insert(kv);
  }

  // load base model
  std::unique_ptr<falcon_model_loader> model_loader;
  ggml_context* base_ctx = NULL;
  llama_ggml::llama_buffer base_buf;
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
      model_loader->mapping.reset(new llama_ggml::llama_mmap(
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

    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&length), sizeof(length));
    fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
    if (fin.eof()) {
      break;
    }

    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
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
    ggml_tensor* lora_tensor;
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
    fin.read((char*)lora_tensor->data, tensor_data_size);

    lora_tensors[name] = lora_tensor;

    // check if we have both A and B tensors and apply
    if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
        lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {
      ggml_tensor* dest_t = model_tensors[base_name];
      ggml_tensor* base_t;
      if (model_loader) {
        // load from base model
        if (model_loader->tensors_map.name_to_idx.find(base_name) ==
            model_loader->tensors_map.name_to_idx.end()) {
          fprintf(stderr, "%s: error: tensor '%s' not found in base model\n",
                  __func__, base_name.c_str());
          return 1;
        }
        size_t idx = model_loader->tensors_map.name_to_idx[base_name];
        falcon_load_tensor& lt = model_loader->tensors_map.tensors[idx];
        base_t = model_loader->get_tensor(
            base_name, {(uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1]},
            GGML_BACKEND_CPU);
        lt.data = (uint8_t*)lt.ggml_tensor->data;
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

      ggml_tensor* loraA = lora_tensors[base_name + ".loraA"];
      ggml_tensor* loraB = lora_tensors[base_name + ".loraB"];

      if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
        fprintf(stderr,
                "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64
                ");"
                " are you sure that this adapter is for this model?\n",
                __func__, base_t->ne[0], loraA->ne[1]);
        return 1;
      }

      // w = w + BA*s
      ggml_tensor* BA = ggml_mul_mat(lora_ctx, loraA, loraB);

      if (scaling != 1.0f) {
        ggml_tensor* scale_tensor = ggml_new_f32(lora_ctx, scaling);
        BA = ggml_scale_inplace(lora_ctx, BA, scale_tensor);
      }

      ggml_tensor* r;
      if (base_t == dest_t) {
        r = ggml_add_inplace(lora_ctx, dest_t, BA);
      } else {
        r = ggml_add(lora_ctx, base_t, BA);
        r = ggml_cpy(lora_ctx, r, dest_t);
      }

      struct ggml_cgraph gf = ggml_build_forward(r);
      ggml_graph_compute_with_ctx(lora_ctx, &gf, n_threads);

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

int falcon_apply_lora_from_file(struct falcon_context* ctx,
                                const char* path_lora,
                                const char* path_base_model, int n_threads) {
  try {
    return falcon_apply_lora_from_file_internal(ctx, path_lora, path_base_model,
                                                n_threads);
  } catch (const std::exception& err) {
    fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__,
            err.what());
    return 1;
  }
}

int falcon_get_kv_cache_token_count(const struct falcon_context* ctx) {
  return ctx->model.kv_self.n;
}

void falcon_set_rng_seed(struct falcon_context* ctx, int seed) {
  if (seed < 0) {
    seed = time(NULL);
  }
  ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t falcon_get_state_size(const struct falcon_context* ctx) {
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
size_t falcon_copy_state_data(struct falcon_context* ctx, uint8_t* dst) {
  uint8_t* out = dst;

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
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;
    const int n_head_kv = hparams.n_head_kv;
    const int head_dim = hparams.n_embd / hparams.n_head;

    const size_t kv_size = kv_self.buf.size;
    const int kv_ntok = falcon_get_kv_cache_token_count(ctx);

    memcpy(out, &kv_size, sizeof(kv_size));
    out += sizeof(kv_size);
    memcpy(out, &kv_ntok, sizeof(kv_ntok));
    out += sizeof(kv_ntok);

    if (kv_size) {
      const size_t elt_size = ggml_element_size(kv_self.k);

      ggml_context* cpy_ctx = ggml_init({4096, NULL, /* no_alloc */ true});
      ggml_cgraph gf{};

      // ggml_tensor * kout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type,
      // n_embd, kv_ntok, n_layer);
      ggml_tensor* kout3d = ggml_new_tensor_3d(
          cpy_ctx, kv_self.k->type, n_head_kv * head_dim, kv_ntok, n_layer);
      kout3d->data = out;
      out += ggml_nbytes(kout3d);

// ggml_tensor * vout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok,
// n_embd, n_layer);
#ifdef FALCON_NO_KV_UPGRADE
      ggml_tensor* vout3d = ggml_new_tensor_3d(
          cpy_ctx, kv_self.v->type, n_head_kv * head_dim, kv_ntok, n_layer);
#else
      ggml_tensor* vout3d = ggml_new_tensor_3d(
          cpy_ctx, kv_self.v->type, kv_ntok, n_head_kv * head_dim, n_layer);
#endif
      vout3d->data = out;
      out += ggml_nbytes(vout3d);

      /*ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
          n_embd, kv_ntok, n_layer,
          elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

      ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
          kv_ntok, n_embd, n_layer,
          elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);*/
      // todo: wouldn't this all be more consistent as 1d ?
      ggml_tensor* k3d =
          ggml_view_3d(cpy_ctx, kv_self.k, n_head_kv * head_dim, kv_ntok,
                       n_layer, elt_size * n_head_kv * head_dim,
                       elt_size * n_head_kv * head_dim * n_ctx, 0);
#ifdef FALCON_NO_KV_UPGRADE
      ggml_tensor* v3d =
          ggml_view_3d(cpy_ctx, kv_self.v, n_head_kv * head_dim, kv_ntok,
                       n_layer, elt_size * n_head_kv * head_dim,
                       elt_size * n_head_kv * head_dim * n_ctx, 0);
#else
      ggml_tensor* v3d = ggml_view_3d(
          cpy_ctx,
          (kv_self.v_current == kv_self.V_A) ? kv_self.v_a : kv_self.v_b,
          kv_ntok, n_head_kv * head_dim, n_layer,
          /*nb1*/ elt_size * kv_ntok,
          /*nb2*/ n_ctx * head_dim * n_head_kv * elt_size,
          /*off*/ 0);

#endif

      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, k3d, kout3d));
      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, v3d, vout3d));
      ggml_graph_compute_with_ctx(cpy_ctx, &gf, /*n_threads=*/1);

      ggml_free(cpy_ctx);
    }
  }

  const size_t written = out - dst;
  const size_t max_size = falcon_get_state_size(ctx);

  LLAMA_ASSERT(written <= max_size);

  return written;
}

// Sets (restores) the state reading from the specified source address
size_t falcon_set_state_data(struct falcon_context* ctx, uint8_t* src) {
  uint8_t* inp = src;

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

  // set/restore kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;
    const int n_head_kv = hparams.n_head_kv;
    const int head_dim = hparams.n_embd / hparams.n_head;

    size_t kv_size;
    int kv_ntok;

    memcpy(&kv_size, inp, sizeof(kv_size));
    inp += sizeof(kv_size);
    memcpy(&kv_ntok, inp, sizeof(kv_ntok));
    inp += sizeof(kv_ntok);

    if (kv_size) {
      LLAMA_ASSERT(kv_self.buf.size == kv_size);

      const size_t elt_size = ggml_element_size(kv_self.k);

      ggml_context* cpy_ctx = ggml_init({4096, NULL, /* no_alloc */ true});
      ggml_cgraph gf{};
      // ggml_tensor * kin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type,
      // n_embd, kv_ntok, n_layer);
      ggml_tensor* kin3d = ggml_new_tensor_3d(
          cpy_ctx, kv_self.k->type, n_head_kv * head_dim, kv_ntok, n_layer);
      kin3d->data = (void*)inp;
      inp += ggml_nbytes(kin3d);

// ggml_tensor * vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok,
// n_embd, n_layer);
#ifdef FALCON_NO_KV_UPGRADE
      ggml_tensor* vin3d = ggml_new_tensor_3d(
          cpy_ctx, kv_self.v->type, n_head_kv * head_dim, kv_ntok, n_layer);
#else
      ggml_tensor* vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok,
                                              n_head_kv * head_dim, n_layer);
#endif
      vin3d->data = (void*)inp;
      inp += ggml_nbytes(vin3d);

      /*ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
          n_embd, kv_ntok, n_layer,
          elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

      ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
          kv_ntok, n_embd, n_layer,
          elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);*/
      ggml_tensor* k3d =
          ggml_view_3d(cpy_ctx, kv_self.k, n_head_kv * head_dim, kv_ntok,
                       n_layer, elt_size * n_head_kv * head_dim,
                       elt_size * n_head_kv * head_dim * n_ctx, 0);

#ifdef FALCON_NO_KV_UPGRADE
      ggml_tensor* v3d =
          ggml_view_3d(cpy_ctx, kv_self.v, n_head_kv * head_dim, kv_ntok,
                       n_layer, elt_size * n_head_kv * head_dim,
                       elt_size * n_head_kv * head_dim * n_ctx, 0);
#else
      ggml_tensor* v3d = ggml_view_3d(
          cpy_ctx,
          (kv_self.v_current == kv_self.V_A) ? kv_self.v_a : kv_self.v_b,
          kv_ntok, n_head_kv * head_dim, n_layer,
          /*nb1*/ elt_size * kv_ntok,
          /*nb2*/ n_ctx * head_dim * n_head_kv * elt_size,
          /*off*/ 0);
#endif

      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, kin3d, k3d));
      ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, vin3d, v3d));
      ggml_graph_compute_with_ctx(cpy_ctx, &gf, /*n_threads=*/1);

      ggml_free(cpy_ctx);
    }

    ctx->model.kv_self.n = kv_ntok;
  }

  const size_t nread = inp - src;
  const size_t max_size = falcon_get_state_size(ctx);

  LLAMA_ASSERT(nread <= max_size);

  return nread;
}

static bool falcon_load_session_file_internal(struct falcon_context* ctx,
                                              const char* path_session,
                                              falcon_token* tokens_out,
                                              size_t n_token_capacity,
                                              size_t* n_token_count_out) {
  llama_ggml::llama_file file(path_session, "rb");

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

    file.read_raw(tokens_out, sizeof(falcon_token) * n_token_count);
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

bool falcon_load_session_file(struct falcon_context* ctx,
                              const char* path_session,
                              falcon_token* tokens_out, size_t n_token_capacity,
                              size_t* n_token_count_out) {
  try {
    return falcon_load_session_file_internal(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
  } catch (const std::exception& err) {
    fprintf(stderr, "error loading session file: %s\n", err.what());
    return false;
  }
}

bool falcon_save_session_file(struct falcon_context* ctx,
                              const char* path_session,
                              const falcon_token* tokens,
                              size_t n_token_count) {
  llama_ggml::llama_file file(path_session, "wb");

  file.write_u32(LLAMA_SESSION_MAGIC);
  file.write_u32(LLAMA_SESSION_VERSION);

  file.write_raw(&ctx->model.hparams, sizeof(falcon_hparams));

  // save the prompt
  file.write_u32((uint32_t)n_token_count);
  file.write_raw(tokens, sizeof(falcon_token) * n_token_count);

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

int falcon_eval(struct falcon_context* ctx, const falcon_token* tokens,
                int n_tokens, int n_past, int n_threads, int debug_timings) {
  //  fprintf(stderr, "falcon_eval: n_tokens=%d, n_past=%d, n_threads=%d\n",
  //  n_tokens, n_past, n_threads);
  // fprintf(stderr, "n_ctx=%d, n_embd=%d, n_head=%d, n_layer=%d, n_vocab=%d\n",
  // ctx->model.hparams.n_ctx, ctx->model.hparams.n_embd,
  // ctx->model.hparams.n_head, ctx->model.hparams.n_layer,
  // ctx->model.hparams.n_vocab);
  LLAMA_ASSERT(ctx->model.hparams.n_ctx >=
               (n_past + n_tokens));  // kv buffer overflow
#if defined(GGML_USE_CUBLAS)
  static int no_purge_counter =
      0;  // once the system is stable for 3 iterations, we stop testing
  if (no_purge_counter < 3 || n_past % 50 == 0) {
    int purges = 0;
    const GPUStatus* status = ggml_cuda_get_system_gpu_status();
    for (int i = 0; i < status->num_devices; i++) {
      purges += ggml_cuda_pool_purge_buffers_with_access_count(1, i);
      ggml_cuda_pool_reset_all_counters(i);
    }
    if (!purges && n_tokens == 1)
      no_purge_counter++;
    else
      no_purge_counter = 0;
  }
#endif

  if (!falcon_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads, nullptr,
                            debug_timings)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }
  // ggml_cuda_update_gpu_status(-1);
  // ggml_cuda_print_gpu_status(ggml_cuda_get_system_gpu_status(),true);
  if (!ctx->has_evaluated_once) {
    ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
    ctx->has_evaluated_once = true;
  }

  return 0;
}

int falcon_eval_export(struct falcon_context* ctx, const char* fname) {
  const int n_batch = 1;
  const int n_ctx = 512 - n_batch;

  const std::vector<falcon_token> tmp(n_batch, falcon_token_bos());

  if (!falcon_eval_internal(*ctx, tmp.data(), tmp.size(), n_ctx, 1, fname, 0)) {
    fprintf(stderr, "%s: failed to eval\n", __func__);
    return 1;
  }

  return 0;
}

int falcon_tokenize(struct falcon_context* ctx, const char* text,
                    falcon_token* tokens, int n_max_tokens, bool add_bos) {
  auto res = falcon_tokenize(ctx->vocab, text, add_bos, true);

  if (n_max_tokens < (int)res.size()) {
    fprintf(stderr, "%s: too many tokens: %d < %zu\n", __func__, n_max_tokens,
            res.size());
    return -((int)res.size());
  }

  for (size_t i = 0; i < res.size(); i++) {
    tokens[i] = res[i];
  }

  return res.size();
}

int falcon_n_vocab(const struct falcon_context* ctx) {
  return ctx->vocab.id_to_token.size();
}

int falcon_n_ctx(const struct falcon_context* ctx) {
  return ctx->model.hparams.n_ctx;
}

int falcon_n_embd(const struct falcon_context* ctx) {
  return ctx->model.hparams.n_embd;
}

int falcon_get_vocab(const struct falcon_context* ctx, const char** strings,
                     float* scores, int capacity) {
  int n = std::min(capacity, (int)ctx->vocab.id_to_token.size());
  for (int i = 0; i < n; ++i) {
    strings[i] = ctx->vocab.id_to_token[i].tok.c_str();
    scores[i] = ctx->vocab.id_to_token[i].score;
  }
  return n;
}

float* falcon_get_logits(struct falcon_context* ctx) {
  return ctx->logits.data();
}

float* falcon_get_embeddings(struct falcon_context* ctx) {
  return ctx->embedding.data();
}

const char* falcon_token_to_str(const struct falcon_context* ctx,
                                falcon_token token) {
  if (token >= falcon_n_vocab(ctx)) {
    return nullptr;
  }

  return ctx->vocab.id_to_token[token].tok.c_str();
}

falcon_token falcon_token_bos() { return 11; }

falcon_token falcon_token_eos() { return 11; }

falcon_token falcon_token_nl() { return 193; }

falcon_token falcon_token_cr() { return 195; }
