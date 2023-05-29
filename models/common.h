#ifndef CTRANSFORMERS_MODELS_COMMON_H_
#define CTRANSFORMERS_MODELS_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ggml/ggml.h"

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

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab &vocab,
                                        const std::string &text) {
  std::vector<std::string> words;

  // first split the text into words
  {
    std::string str = text;
    std::string pat =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    // Generate the subpattern from the special_tokens vector if it's not empty
    if (!vocab.special_tokens.empty()) {
      std::string special_tokens_subpattern;
      for (const auto &token : vocab.special_tokens) {
        if (!special_tokens_subpattern.empty()) {
          special_tokens_subpattern += "|";
        }
        special_tokens_subpattern += token;
      }

      // Modify the regex pattern with the generated special tokens subpattern
      pat = special_tokens_subpattern + "|" + pat;
    }

    std::regex re(pat);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
      for (auto x : m) {
        words.push_back(x);
      }
      str = m.suffix();
    }
  }

  // find the longest tokens that form the words:
  std::vector<gpt_vocab::id> tokens;
  for (const auto &word : words) {
    if (word.size() == 0) continue;

    int i = 0;
    int n = word.size();
    while (i < n) {
      int j = n;
      while (j > i) {
        auto it = vocab.token_to_id.find(word.substr(i, j - i));
        if (it != vocab.token_to_id.end()) {
          tokens.push_back(it->second);
          i = j;
          j = n;
          break;
        }
        --j;
      }
      if (i == n) {
        break;
      }
      if (j == i) {
        auto sub = word.substr(i, 1);
        if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
          tokens.push_back(vocab.token_to_id.at(sub));
        } else {
          fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
        }
        ++i;
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

#endif
