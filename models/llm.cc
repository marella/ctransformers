#include "llm.h"

#include "llms/dolly.cc"
#include "llms/gpt-neox.cc"
#include "llms/gpt2.cc"
#include "llms/gptj.cc"
#include "llms/llama.cc"
#include "llms/mpt.cc"
#include "llms/replit.cc"
#include "llms/starcoder.cc"

// Old ggml file format.
#include "llms/llama-ggml.cc"
// Import falcon after llama.
#include "llms/falcon.cc"

bool is_gguf(const char* filename) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return false;
  }
  uint32_t magic = 0;
  const size_t size = sizeof(magic);
  const size_t read = fread(&magic, 1, size, file);
  fclose(file);
  if (read != size) {
    return false;
  }
  return magic == GGUF_MAGIC;
}

#ifdef __cplusplus
extern "C" {
#endif

LLM* ctransformers_llm_create(const char* model_path, const char* model_type,
                              const Config config) {
  std::string type = model_type;
  // Remove non-alphanumeric characters from model type.
  type.erase(std::remove_if(type.begin(), type.end(),
                            [](const char c) { return !std::isalnum(c); }),
             type.end());

  LLM* llm = nullptr;
  if (type == "gguf" || is_gguf(model_path)) {
    llm = new llama_llm;
  } else if (type == "dollyv2") {
    llm = new dollyv2_llm;
  } else if (type == "falcon") {
    llm = new falcon_llm;
  } else if (type == "gpt2") {
    llm = new gpt2_llm;
  } else if (type == "gptj") {
    llm = new gptj_llm;
  } else if (type == "gptneox") {
    llm = new gpt_neox_llm;
  } else if (type == "llama") {
    llm = new llama_ggml::llama_llm;
  } else if (type == "mpt") {
    llm = new mpt_llm;
  } else if (type == "replit") {
    llm = new replit_llm;
  } else if (type == "starcoder" || type == "gptbigcode") {
    llm = new starcoder_llm;
  }

  if (llm == nullptr) {
    fprintf(stderr, "Model type '%s' is not supported.\n", model_type);
    return nullptr;
  }
  if (!llm->Init(model_path, config)) {
    delete llm;
    return nullptr;
  }
  return llm;
}

void ctransformers_llm_delete(LLM* llm) { delete llm; }

int ctransformers_llm_tokenize(LLM* llm, const char* text,
                               const bool add_bos_token, int* output) {
  const std::vector<gpt_vocab::id> tokens = llm->Tokenize(text, add_bos_token);
  std::copy(tokens.begin(), tokens.end(), output);
  return tokens.size();
}

const char* ctransformers_llm_detokenize(LLM* llm, const int token) {
  return llm->Detokenize(token).c_str();
}

bool ctransformers_llm_is_eos_token(LLM* llm, const int token) {
  return llm->IsEosToken(token);
}

int ctransformers_llm_eos_token_id(LLM* llm) { return llm->EosToken(); }

int ctransformers_llm_bos_token_id(LLM* llm) { return llm->BosToken(); }

int ctransformers_llm_vocab_size(LLM* llm) { return llm->VocabSize(); }

int ctransformers_llm_context_length(LLM* llm) { return llm->ContextLength(); }

const char* ctransformers_llm_architecture(LLM* llm) {
  return llm->Architecture().c_str();
}

bool ctransformers_llm_batch_eval(LLM* llm, const int* tokens,
                                  const int n_tokens, const int n_past,
                                  const int batch_size, const int threads) {
  return llm->BatchEval(std::vector<gpt_vocab::id>(tokens, tokens + n_tokens),
                        n_past, batch_size, threads);
}

float* ctransformers_llm_logits_data(LLM* llm) { return llm->Logits().data(); }

int ctransformers_llm_logits_size(LLM* llm) { return llm->Logits().size(); }

const float* ctransformers_llm_embeddings_data(LLM* llm) {
  return llm->Embeddings().data();
}

int ctransformers_llm_embeddings_size(LLM* llm) {
  return llm->Embeddings().size();
}

int ctransformers_llm_sample(LLM* llm, const int* last_tokens, const int n_last,
                             const int top_k, const float top_p,
                             const float temperature,
                             const float repetition_penalty, const int seed) {
  return llm->Sample(last_tokens, n_last, top_k, top_p, temperature,
                     repetition_penalty, seed);
}

void ctransformers_llm_reset(LLM* llm) { llm->Reset(); }

#ifdef __cplusplus
}
#endif
