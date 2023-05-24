#include <cstdio>
#include <vector>

extern "C" {
struct LLM;

LLM* ctransformers_llm_create(const char* model_path, const char* model_type);

void ctransformers_llm_delete(LLM* llm);

int ctransformers_llm_tokenize(LLM* llm, const char* text, int* output);

const char* ctransformers_llm_detokenize(LLM* llm, const int token);

bool ctransformers_llm_batch_eval(LLM* llm, const int* tokens,
                                  const int n_tokens, const int batch_size,
                                  const int threads);

int ctransformers_llm_sample(LLM* llm, const int top_k, const float top_p,
                             const float temperature,
                             const float repetition_penalty,
                             const int last_n_tokens, const int seed);
}

int main(const int argc, const char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: main <model_type> <model_path>\n");
    return 1;
  }

  const char* model_type = argv[1];
  const char* model_path = argv[2];
  const char* prompt = "Hi";
  printf("\n");
  printf("model type : '%s'\n", model_type);
  printf("model path : '%s'\n", model_path);
  printf("prompt     : '%s'\n", prompt);
  printf("\n");

  const int top_k = 40;
  const float top_p = 0.95;
  const float temperature = 0.8;
  const float repetition_penalty = 1.1;
  const int last_n_tokens = 64;
  const int seed = -1;
  const int batch_size = 8;
  const int threads = -1;

  printf("load ... ");
  fflush(stdout);
  LLM* llm = ctransformers_llm_create(model_path, model_type);
  if (llm == nullptr) {
    printf("🗙\n\n");
    return 1;
  }
  printf("✔\n");

  printf("tokenize ... ");
  fflush(stdout);
  std::vector<int> tokens(10);
  const int n_tokens = ctransformers_llm_tokenize(llm, prompt, tokens.data());
  tokens.resize(n_tokens);
  printf("✔\n");
  printf("> [ ");
  for (const int token : tokens) {
    printf("%d ", token);
  }
  printf("]\n");

  printf("eval ... ");
  fflush(stdout);
  const bool status = ctransformers_llm_batch_eval(
      llm, tokens.data(), tokens.size(), batch_size, threads);
  if (!status) {
    printf("🗙\n\n");
    return 1;
  }
  printf("✔\n");

  printf("sample ... ");
  fflush(stdout);
  const int token = ctransformers_llm_sample(
      llm, top_k, top_p, temperature, repetition_penalty, last_n_tokens, seed);
  printf("✔\n");
  printf("> %d\n", token);

  printf("detokenize ... ");
  fflush(stdout);
  const char* text = ctransformers_llm_detokenize(llm, token);
  printf("✔\n");
  printf("> '%s'\n", text);

  printf("delete ... ");
  fflush(stdout);
  ctransformers_llm_delete(llm);
  printf("✔\n");

  printf("\n");

  return 0;
}
