#include "llm.h"

// https://github.com/cmp-nct/ggllm.cpp/blob/master/examples/falcon/falcon_main.cpp

#include "ggml/libfalcon.cpp"

class falcon_llm : public LLM {
 public:
  virtual ~falcon_llm() {
    if (ctx_ != nullptr) {
      falcon_free(ctx_);
    }
  }

  std::vector<gpt_vocab::id> Tokenize(const std::string &text,
                                      const bool add_bos_token) const override {
    return falcon_tokenize(ctx_->vocab, text, /*bos=*/false, /*g2ws=*/true);
  }

  const std::string &Detokenize(const gpt_vocab::id id) const override {
    if (id >= falcon_n_vocab(ctx_)) {
      return kEmptyString;
    }
    return ctx_->vocab.id_to_token[id].tok;
  }

  bool IsEosToken(const gpt_vocab::id token) const override {
    return token == EosToken();
  }

  gpt_vocab::id EosToken() const override { return falcon_token_eos(); }

  int VocabSize() const override { return falcon_n_vocab(ctx_); }

  std::vector<float> &Logits() override { return ctx_->logits; }

  const std::vector<float> &Embeddings() const override {
    return ctx_->embedding;
  }

  gpt_vocab::id Sample(const int top_k, const float top_p,
                       const float temperature, const float repetition_penalty,
                       int last_n_tokens, int seed) const override {
    if (last_n_tokens < 0) {
      last_n_tokens = ContextLength();
    }
    if (seed < 0) {
      seed = time(nullptr);
    }
    ctx_->rng.seed(seed);

    const float *logits = falcon_get_logits(ctx_);
    const int n_vocab = falcon_n_vocab(ctx_);

    std::vector<falcon_token_data> candidates;
    candidates.reserve(n_vocab);
    for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(
          falcon_token_data{token_id, logits[token_id], 0.0f});
    }

    falcon_token_data_array candidates_p = {
        candidates.data(),
        candidates.size(),
        false,
    };

    {
      std::unordered_set<gpt_vocab::id> recent_tokens_set;
      if (repetition_penalty != 1.0f) {
        recent_tokens_set = previous_tokens_.GetRecent(last_n_tokens);
      }
      std::vector<gpt_vocab::id> recent_tokens(recent_tokens_set.begin(),
                                               recent_tokens_set.end());
      falcon_sample_repetition_penalty(
          ctx_, &candidates_p, recent_tokens.data(), recent_tokens.size(),
          repetition_penalty);
    }

    falcon_sample_top_k(ctx_, &candidates_p, top_k, 1);
    falcon_sample_top_p(ctx_, &candidates_p, top_p, 1);
    falcon_sample_temperature(ctx_, &candidates_p, temperature);
    return falcon_sample_token(ctx_, &candidates_p);
  }

 protected:
  bool Load(const std::string &filename, const Config &config) override {
    falcon_context_params params = falcon_context_default_params();
    params.embedding = true;
    if (config.context_length > 0) {
      params.n_ctx = config.context_length;
    }
    params.n_gpu_layers = config.gpu_layers;
    params.use_mmap = config.mmap;
    params.use_mlock = config.mlock;

    ctx_ = falcon_init_from_file(filename.c_str(), params);
    if (ctx_ == nullptr) {
      return false;
    }
    n_ctx_ = falcon_n_ctx(ctx_);
    return true;
  }

  bool Eval(const std::vector<gpt_vocab::id> &tokens, const int threads,
            const int n_past) override {
    const int status = falcon_eval(ctx_, tokens.data(), tokens.size(), n_past,
                                   threads, /*debug_timings=*/0);
    return status == 0;
  }

 private:
  falcon_context *ctx_ = nullptr;
};
