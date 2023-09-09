#ifndef CTRANSFORMERS_MODELS_LLM_H_
#define CTRANSFORMERS_MODELS_LLM_H_

#include "common.h"

struct Config {
  int context_length;
  int gpu_layers;
  bool mmap;
  bool mlock;
};

class LLM {
 public:
  virtual ~LLM(){};

  bool Init(const std::string &filename, const Config &config) {
    if (initialized_) {
      return false;
    }
    if (!Load(filename, config)) {
      return false;
    }
    return initialized_ = true;
  }

  virtual std::vector<gpt_vocab::id> Tokenize(const std::string &text,
                                              const bool add_bos_token) const {
    return gpt_tokenize(vocab_, text);
  }

  virtual const std::string &Detokenize(const gpt_vocab::id id) const {
    const auto it = vocab_.id_to_token.find(id);
    if (it == vocab_.id_to_token.end()) {
      return kEmptyString;
    }
    return it->second;
  }

  bool BatchEval(const std::vector<gpt_vocab::id> &tokens, int n_past,
                 int batch_size, const int threads) {
    batch_size = std::min(ContextLength(), batch_size);
    const int size = tokens.size();
    for (int start = 0; start < size; start += batch_size) {
      const int end = std::min(start + batch_size, (int)tokens.size());
      const std::vector<gpt_vocab::id> batch(tokens.begin() + start,
                                             tokens.begin() + end);
      if (!EvalInternal(batch, n_past, threads)) {
        return false;
      }
      n_past += batch.size();
    }
    return true;
  }

  virtual std::vector<float> &Logits() { return logits_; }

  virtual const std::vector<float> &Embeddings() const { return embeddings_; }

  virtual gpt_vocab::id Sample(const int *last_tokens, const int n_last,
                               const int top_k, const float top_p,
                               const float temperature,
                               const float repetition_penalty, int seed) const {
    if (logits_.empty()) {
      return EosToken();
    }
    if (seed < 0) {
      seed = time(nullptr);
    }
    std::mt19937 rng(seed);
    return gpt_sample_top_k_top_p(
        vocab_, logits_.data() + (logits_.size() - VocabSize()), top_k, top_p,
        temperature, repetition_penalty,
        std::unordered_set<gpt_vocab::id>(last_tokens, last_tokens + n_last),
        rng);
  }

  virtual bool IsEosToken(const gpt_vocab::id token) const {
    if (token == EosToken()) {
      return true;
    }
    // Handle special tokens in StarChat and Dolly V2.
    if (!vocab_.special_tokens.empty()) {
      const std::string &text = Detokenize(token);
      return text == "<|end|>" || text == "### End";
    }
    return false;
  }

  virtual gpt_vocab::id EosToken() const {
    const auto it = vocab_.token_to_id.find("<|endoftext|>");
    if (it != vocab_.token_to_id.end()) {
      return it->second;
    }
    return 0;
  }

  virtual gpt_vocab::id BosToken() const { return EosToken(); }

  virtual int VocabSize() const { return vocab_.id_to_token.size(); }

  int ContextLength() const { return n_ctx_; }

  virtual const std::string &Architecture() const { return kEmptyString; }

  void Reset() { logits_.clear(); }

 protected:
  const std::string kEmptyString = "";
  int n_ctx_ = -1;
  gpt_vocab vocab_;
  size_t mem_per_token_ = 0;
  std::vector<float> logits_;
  std::vector<float> embeddings_;

  virtual bool Load(const std::string &filename, const Config &config) = 0;

  virtual bool Eval(const std::vector<gpt_vocab::id> &tokens, const int threads,
                    const int n_past) = 0;

 private:
  bool initialized_ = false;

  bool EvalInternal(const std::vector<gpt_vocab::id> &tokens, int n_past,
                    int threads) {
    n_past = std::min(ContextLength() - (int)tokens.size(), n_past);
    if (threads < 0) {
      // https://github.com/ggerganov/llama.cpp/blob/cc45a7feb8412e84ff292207621412fffc0d3d51/examples/common.cpp#L67-L68
      const int n = std::thread::hardware_concurrency();
      threads = n > 0 ? (n <= 4 ? n : n / 2) : 4;
    }
    threads = std::max(threads, 1);
    if (!Eval(tokens, threads, n_past)) {
      return false;
    }
    return true;
  }
};

#define REGISTER_LLM(_name)                                                 \
  class _name##_llm : public LLM {                                          \
   public:                                                                  \
    virtual ~_name##_llm() {                                                \
      if (model_.ctx != nullptr) {                                          \
        ggml_free(model_.ctx);                                              \
      }                                                                     \
    }                                                                       \
                                                                            \
   protected:                                                               \
    bool Load(const std::string &filename, const Config &config) override { \
      if (config.context_length > 0) {                                      \
        model_.hparams.n_ctx = config.context_length;                       \
      }                                                                     \
      if (!_name##_model_load(filename, model_, vocab_)) {                  \
        return false;                                                       \
      }                                                                     \
      n_ctx_ = model_.hparams.n_ctx;                                        \
      return true;                                                          \
    }                                                                       \
                                                                            \
    bool Eval(const std::vector<gpt_vocab::id> &tokens, const int threads,  \
              const int n_past) override {                                  \
      return _name##_eval(model_, threads, n_past, tokens, logits_,         \
                          mem_per_token_);                                  \
    }                                                                       \
                                                                            \
   private:                                                                 \
    _name##_model model_;                                                   \
  }

#endif
