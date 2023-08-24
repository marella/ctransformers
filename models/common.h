#ifndef CTRANSFORMERS_MODELS_COMMON_H_
#define CTRANSFORMERS_MODELS_COMMON_H_

#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <locale>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml/ggml.h"

#ifdef GGML_USE_CUBLAS
#include "ggml/ggml-cuda.h"
#endif

// https://github.com/ggerganov/ggml/blob/master/examples/common.cpp

struct gpt_vocab {
  using id = int32_t;
  using token = std::string;

  std::map<token, id> token_to_id;
  std::map<id, token> id_to_token;
  std::vector<std::string> special_tokens;

  void add_special_token(const std::string &token) {
    special_tokens.push_back(token);
  }
};

std::string convert_to_utf8(const std::wstring &input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(input);
}

std::wstring convert_to_wstring(const std::string &input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(input);
}

void gpt_split_words(std::string str, std::vector<std::string> &words) {
  const std::string pattern =
      R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
  const std::regex re(pattern);
  std::smatch m;

  while (std::regex_search(str, m, re)) {
    for (auto x : m) {
      words.push_back(x);
    }
    str = m.suffix();
  }
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab &vocab,
                                        const std::string &text) {
  std::vector<std::string> words;

  // first split the text into words
  {
    std::string str = text;

    // Generate the subpattern from the special_tokens vector if it's not empty
    if (!vocab.special_tokens.empty()) {
      const std::regex escape(R"([\[\\\^\$\.\|\?\*\+\(\)\{\}])");
      std::string special_tokens_subpattern;
      for (const auto &token : vocab.special_tokens) {
        if (!special_tokens_subpattern.empty()) {
          special_tokens_subpattern += "|";
        }
        special_tokens_subpattern +=
            std::regex_replace(token, escape, R"(\$&)");
      }

      std::regex re(special_tokens_subpattern);
      std::smatch m;
      // Split the text by special tokens.
      while (std::regex_search(str, m, re)) {
        // Split the substrings in-between special tokens into words.
        gpt_split_words(m.prefix(), words);
        // Add matched special tokens as words.
        for (auto x : m) {
          words.push_back(x);
        }
        str = m.suffix();
      }
      // Remaining text without special tokens will be handled below.
    }

    gpt_split_words(str, words);
  }

  // find the longest token that forms each word in words:
  std::vector<gpt_vocab::id> tokens;
  for (const auto &word : words) {
    for (int i = 0; i < (int)word.size();) {
      for (int j = word.size() - 1; j >= i; j--) {
        auto cand = word.substr(i, j - i + 1);
        auto it = vocab.token_to_id.find(cand);
        if (it != vocab.token_to_id.end()) {  // word.substr(i, j-i+1) in vocab
          tokens.push_back(it->second);
          i = j + 1;
          break;
        } else if (j == i) {  // word.substr(i, 1) has no matching
          fprintf(stderr, "%s: unknown token '%s'\n", __func__,
                  word.substr(i, 1).data());
          i++;
        }
      }
    }
  }

  return tokens;
}

gpt_vocab::id gpt_sample_top_k_top_p(
    const gpt_vocab &vocab, const float *logits, int top_k, double top_p,
    double temp, const float repetition_penalty,
    const std::unordered_set<gpt_vocab::id> &recent_tokens, std::mt19937 &rng) {
  int n_logits = vocab.id_to_token.size();

  std::vector<std::pair<double, gpt_vocab::id>> logits_id;
  logits_id.reserve(n_logits);

  {
    const double scale = 1.0 / temp;
    for (int i = 0; i < n_logits; ++i) {
      logits_id.push_back(std::make_pair(logits[i] * scale, i));
    }
  }

  for (const gpt_vocab::id token : recent_tokens) {
    // https://github.com/ggerganov/llama.cpp/blob/3e5aa8a1c44051153d6d7b3eeca2f4b4e5fb310c/llama.cpp#L1690-L1717
    // https://github.com/ggerganov/llama.cpp/blob/3e5aa8a1c44051153d6d7b3eeca2f4b4e5fb310c/examples/main/main.cpp#L432-L434
    double &logit = logits_id[token].first;
    if (logit <= 0) {
      logit *= repetition_penalty;
    } else {
      logit /= repetition_penalty;
    }
  }

  // find the top K tokens
  std::partial_sort(logits_id.begin(), logits_id.begin() + top_k,
                    logits_id.end(),
                    [](const std::pair<double, gpt_vocab::id> &a,
                       const std::pair<double, gpt_vocab::id> &b) {
                      return a.first > b.first;
                    });

  logits_id.resize(top_k);

  double maxl = -INFINITY;
  for (const auto &kv : logits_id) {
    maxl = std::max(maxl, kv.first);
  }

  // compute probs for the top K tokens
  std::vector<double> probs;
  probs.reserve(logits_id.size());

  double sum = 0.0;
  for (const auto &kv : logits_id) {
    double p = exp(kv.first - maxl);
    probs.push_back(p);
    sum += p;
  }

  // normalize the probs
  for (auto &p : probs) {
    p /= sum;
  }

  if (top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        top_k = i + 1;
        probs.resize(top_k);
        logits_id.resize(top_k);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(rng);

  return logits_id[idx].second;
}

// Temporary

struct ggml_tensor *ggml_norm(struct ggml_context *ctx, struct ggml_tensor *a) {
  return ggml_norm(ctx, a, 1e-5f);
}

// CUDA

// https://github.com/ggerganov/llama.cpp/blob/332311234a0aa2974b2450710e22e09d90dd6b0b/llama.cpp#L719-L740
ggml_tensor *ct_new_tensor(ggml_context *ctx, enum ggml_type type,
                           const std::vector<int64_t> &shape, const bool gpu) {
  const bool no_alloc = ggml_get_no_alloc(ctx);
  ggml_set_no_alloc(ctx, gpu);
  ggml_tensor *tensor;
  if (shape.size() == 1) {
    tensor = ggml_new_tensor_1d(ctx, type, shape[0]);
  } else if (shape.size() == 2) {
    tensor = ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
  } else {
    GGML_ASSERT(false && "Invalid tensor shape.");
  }
  if (gpu) {
    tensor->backend = GGML_BACKEND_GPU;
  }
  ggml_set_no_alloc(ctx, no_alloc);
  return tensor;
}

ggml_tensor *ct_new_tensor(ggml_context *ctx, enum ggml_type type, int64_t x,
                           const bool gpu) {
  return ct_new_tensor(ctx, type, {x}, gpu);
}

ggml_tensor *ct_new_tensor(ggml_context *ctx, enum ggml_type type, int64_t x,
                           int64_t y, const bool gpu) {
  return ct_new_tensor(ctx, type, {x, y}, gpu);
}

// https://github.com/ggerganov/llama.cpp/blob/332311234a0aa2974b2450710e22e09d90dd6b0b/llama.cpp#L772-L778
uint8_t *ct_alloc(ggml_tensor *tensor) {
  if (tensor->backend == GGML_BACKEND_CPU) {
    return (uint8_t *)tensor->data;
  }
  return (uint8_t *)malloc(ggml_nbytes(tensor));
}

// https://github.com/ggerganov/llama.cpp/blob/332311234a0aa2974b2450710e22e09d90dd6b0b/llama.cpp#L783-L797
void ct_transform(uint8_t *data, ggml_tensor *tensor) {
  if (tensor->backend == GGML_BACKEND_CPU) {
    return;
  }
#ifdef GGML_USE_CUBLAS
  ggml_cuda_transform_tensor(data, tensor);
#else
  GGML_ASSERT(false && "CUDA is not enabled.");
#endif
  free(data);
}

// https://github.com/ggerganov/llama.cpp/blob/332311234a0aa2974b2450710e22e09d90dd6b0b/llama.cpp#L321-L325
void ct_free(const std::map<std::string, struct ggml_tensor *> &tensors) {
#ifdef GGML_USE_CUBLAS
  for (const auto &kv : tensors) {
    ggml_cuda_free_data(kv.second);
  }
  ggml_cuda_free_scratch();
#endif
}

#endif
