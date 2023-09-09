#include "llm.h"

// https://github.com/ggerganov/llama.cpp/blob/master/examples/main/main.cpp

#include "ggml/llama-ggml.cpp"

namespace llama_ggml {

class llama_llm : public LLM {
 public:
  virtual ~llama_llm() {
    if (ctx_ != nullptr) {
      llama_free(ctx_);
    }
  }

  std::vector<gpt_vocab::id> Tokenize(const std::string &text,
                                      const bool add_bos_token) const override {
    return llama_tokenize(ctx_->model.vocab, text, add_bos_token);
  }

  const std::string &Detokenize(const gpt_vocab::id id) const override {
    if (id >= llama_n_vocab(ctx_)) {
      return kEmptyString;
    }
    return ctx_->model.vocab.id_to_token[id].tok;
  }

  bool IsEosToken(const gpt_vocab::id token) const override {
    return token == EosToken();
  }

  gpt_vocab::id EosToken() const override { return llama_token_eos(); }

  gpt_vocab::id BosToken() const override { return llama_token_bos(); }

  int VocabSize() const override { return llama_n_vocab(ctx_); }

  std::vector<float> &Logits() override { return ctx_->logits; }

  const std::vector<float> &Embeddings() const override {
    return ctx_->embedding;
  }

  gpt_vocab::id Sample(const int *last_tokens, const int n_last,
                       const int top_k, const float top_p,
                       const float temperature, const float repetition_penalty,
                       int seed) const override {
    if (seed < 0) {
      seed = time(nullptr);
    }
    ctx_->rng.seed(seed);

    const float *logits = llama_get_logits(ctx_);
    const int n_vocab = llama_n_vocab(ctx_);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(
          llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = {
        candidates.data(),
        candidates.size(),
        false,
    };

    llama_sample_repetition_penalty(ctx_, &candidates_p, last_tokens, n_last,
                                    repetition_penalty);
    llama_sample_top_k(ctx_, &candidates_p, top_k, 1);
    llama_sample_top_p(ctx_, &candidates_p, top_p, 1);
    llama_sample_temperature(ctx_, &candidates_p, temperature);
    return llama_sample_token(ctx_, &candidates_p);
  }

 protected:
  bool Load(const std::string &filename, const Config &config) override {
    llama_context_params params = llama_context_default_params();
    params.embedding = true;
    if (config.context_length > 0) {
      params.n_ctx = config.context_length;
    }
    params.n_gpu_layers = config.gpu_layers;
    params.use_mmap = config.mmap;
    params.use_mlock = config.mlock;
    std::regex pattern_70b(R"((\b|_)70b(\b|_))", std::regex_constants::icase);
    if (std::regex_search(filename, pattern_70b)) {
      params.n_gqa = 8;
    }

    llama_model *model = llama_load_model_from_file(filename.c_str(), params);
    ctx_ = llama_new_context_with_model(model, params);
    if (ctx_ == nullptr) {
      return false;
    }
    ctx_->model_owner = true;
    n_ctx_ = llama_n_ctx(ctx_);
    return true;
  }

  bool Eval(const std::vector<gpt_vocab::id> &tokens, const int threads,
            const int n_past) override {
    const int status =
        llama_eval(ctx_, tokens.data(), tokens.size(), n_past, threads);
    return status == 0;
  }

 private:
  llama_context *ctx_ = nullptr;
};

}  // namespace llama_ggml
