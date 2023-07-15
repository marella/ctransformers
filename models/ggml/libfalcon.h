#ifndef FALCON_H
#define FALCON_H

#include "ggml.h"
#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#define LLAMA_MAX_DEVICES GGML_CUDA_MAX_DEVICES
#else
#define LLAMA_MAX_DEVICES 1
#endif  // GGML_USE_CUBLAS
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <string>

#ifdef LLAMA_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef LLAMA_BUILD
#define LLAMA_API __declspec(dllexport)
#else
#define LLAMA_API __declspec(dllimport)
#endif
#else
#define LLAMA_API __attribute__((visibility("default")))
#endif
#else
#define LLAMA_API
#endif

#define LLAMA_FILE_MAGIC_GGJT 0x67676a74u  // 'ggjt'
#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u  // 'ggla'
#define LLAMA_FILE_MAGIC_GGMF 0x67676d66u  // 'ggmf'
#define LLAMA_FILE_MAGIC_GGML 0x67676d6cu  // 'ggml'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu  // 'ggsn'

#define FALCON_FILE_MAGIC_GGCC \
  0x67676363u  // 'ggcc' (cmp-cnt enhancements for ggllm.cpp)

#define FALCON_FILE_VERSION FALCON_FILE_VERSION_GGCC_V1
#define FALCON_FILE_MAGIC FALCON_FILE_MAGIC_GGCC
#define LLAMA_FILE_MAGIC_UNVERSIONED LLAMA_FILE_MAGIC_GGML
#define LLAMA_SESSION_MAGIC LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 1

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST) || \
    defined(GGML_USE_METAL)
// Defined when llama.cpp is compiled with support for offloading model layers
// to GPU.
#define LLAMA_SUPPORTS_GPU_OFFLOAD
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// C interface
//
// TODO: show sample usage
//

struct falcon_context;
struct falcon_model;
struct falcon_vocab;

typedef int falcon_token;

typedef struct falcon_token_data {
  falcon_token id;  // token id
  float logit;      // log-odds of the token
  float p;          // probability of the token
} falcon_token_data;

typedef struct falcon_token_data_array {
  falcon_token_data *data;
  size_t size;
  bool sorted;
} falcon_token_data_array;

typedef void (*falcon_progress_callback)(float progress, void *ctx,
                                         const char *status);

struct falcon_context_params {
  int n_ctx;         // text context
  int n_batch;       // prompt processing batch size
  int n_gpu_layers;  // number of layers to store in VRAM
  int i_gpu_start;   // first gpu layer
  int i_gpu_last;    // last gpu layer
  int main_gpu;      // the GPU that is used for scratch and small tensors
  float tensor_split[LLAMA_MAX_DEVICES];  // how to split layers across multiple
                                          // GPUs
  int seed;                               // RNG seed, -1 for random

  bool f16_kv;      // use fp16 for KV cache
  bool logits_all;  // the llama_eval() call computes all logits, not just the
                    // last one
  bool vocab_only;  // only load the vocabulary, no weights
  bool use_mmap;    // use mmap if possible
  bool use_mlock;   // force system to keep model in RAM
  bool embedding;   // embedding mode only

  // called with a progress value between 0 and 1, pass NULL to disable
  falcon_progress_callback progress_callback;
  // context pointer passed to the progress callback
  void *progress_callback_user_data;
};

// model quantization parameters
typedef struct falcon_model_quantize_params {
  int nthread;  // number of threads to use for quantizing, if <=0 will use
                // std::thread::hardware_concurrency()
  enum llama_ftype ftype;       // quantize to this llama_ftype
  bool allow_requantize;        // allow quantizing non-f32/f16 tensors
  bool quantize_output_tensor;  // quantize output.weight
} falcon_model_quantize_params;

LLAMA_API struct falcon_context_params falcon_context_default_params();
LLAMA_API struct falcon_model_quantize_params
falcon_model_quantize_default_params();

// Various functions for loading a ggml llama model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure
LLAMA_API struct falcon_context *falcon_init_from_file(
    const char *path_model, struct falcon_context_params params);

// prepare scratch and computation buffers
LLAMA_API void falcon_context_set_buffers(falcon_context *ctx, int n_batch,
                                          int n_ctx);
LLAMA_API struct falcon_model *falcon_get_falcon_model(falcon_context *ctx);
// Frees all allocated memory
LLAMA_API void falcon_free(struct falcon_context *ctx);

// Returns 0 on success
LLAMA_API int falcon_model_quantize(const char *fname_inp,
                                    const char *fname_out,
                                    const falcon_model_quantize_params *params);

// Apply a LoRA adapter to a loaded model
// path_base_model is the path to a higher quality model to use as a base for
// the layers modified by the adapter. Can be NULL to use the current loaded
// model. The model needs to be reloaded before applying a new adapter,
// otherwise the adapter will be applied on top of the previous one Returns 0 on
// success
LLAMA_API int falcon_apply_lora_from_file(struct falcon_context *ctx,
                                          const char *path_lora,
                                          const char *path_base_model,
                                          int n_threads);

// Returns the number of tokens in the KV cache
LLAMA_API int falcon_get_kv_cache_token_count(const struct falcon_context *ctx);

// Sets the current rng seed.
LLAMA_API void falcon_set_rng_seed(struct falcon_context *ctx, int seed);

// Returns the maximum size in bytes of the state (rng, logits, embedding
// and kv_cache) - will often be smaller after compacting tokens
LLAMA_API size_t falcon_get_state_size(const struct falcon_context *ctx);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
LLAMA_API size_t falcon_copy_state_data(struct falcon_context *ctx,
                                        uint8_t *dst);

// Set the state reading from the specified address
// Returns the number of bytes read
LLAMA_API size_t falcon_set_state_data(struct falcon_context *ctx,
                                       uint8_t *src);

// Save/load session file
LLAMA_API bool falcon_load_session_file(struct falcon_context *ctx,
                                        const char *path_session,
                                        falcon_token *tokens_out,
                                        size_t n_token_capacity,
                                        size_t *n_token_count_out);
LLAMA_API bool falcon_save_session_file(struct falcon_context *ctx,
                                        const char *path_session,
                                        const falcon_token *tokens,
                                        size_t n_token_count);

// Run the llama inference to obtain the logits and probabilities for the next
// token. tokens + n_tokens is the provided batch of new tokens to process
// n_past is the number of tokens to use from previous eval calls
// Returns 0 on success
LLAMA_API int falcon_eval(struct falcon_context *ctx,
                          const falcon_token *tokens, int n_tokens, int n_past,
                          int n_threads, int debug_timings);

// Export a static computation graph for context of 511 and batch size of 1
// NOTE: since this functionality is mostly for debugging and demonstration
// purposes, we hardcode these
//       parameters here to keep things simple
// IMPORTANT: do not use for anything else other than debugging and testing!
LLAMA_API int falcon_eval_export(struct falcon_context *ctx, const char *fname);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns a negative number on failure - the number of tokens that would have
// been returned
// TODO: not sure if correct
LLAMA_API int falcon_tokenize(struct falcon_context *ctx, const char *text,
                              falcon_token *tokens, int n_max_tokens,
                              bool add_bos);

LLAMA_API int falcon_n_vocab(const struct falcon_context *ctx);
LLAMA_API int falcon_n_ctx(const struct falcon_context *ctx);
LLAMA_API int falcon_n_embd(const struct falcon_context *ctx);

// Get the vocabulary as output parameters.
// Returns number of results.
LLAMA_API int falcon_get_vocab(const struct falcon_context *ctx,
                               const char **strings, float *scores,
                               int capacity);

// prepares a falcon_context based on a model, also allocates scratch buffers
// based on parameters
LLAMA_API struct falcon_context *falcon_context_prepare(
    falcon_context_params params, falcon_model *model, std::string context_name,
    bool verbose);

// Token logits obtained from the last call to llama_eval()
// The logits for the last token are stored in the last row
// Can be mutated in order to change the probabilities of the next token
// Rows: n_tokens
// Cols: n_vocab
LLAMA_API float *falcon_get_logits(struct falcon_context *ctx);

// Get the embeddings for the input
// shape: [n_embd] (1-dimensional)
LLAMA_API float *falcon_get_embeddings(struct falcon_context *ctx);

// Token Id -> String. Uses the vocabulary in the provided context
LLAMA_API const char *falcon_token_to_str(const struct falcon_context *ctx,
                                          falcon_token token);
typedef enum {
  FINETUNE_UNSPECIFIED,
  FINETUNE_NONE,
  FINETUNE_ALPACA,
  FINETUNE_OPENASSISTANT,
  FINETUNE_OPENASSIST_V1,
  FINETUNE_WIZARD,
  FINETUNE_FALCONINSTRUCT
} t_finetune_type;
static const char *FINETUNE_NAME[7] = {
    "UNSPECIFIED",   "NONE",   "ALPACA",        "OPENASSISTANT",
    "OPENASSIST_V1", "WIZARD", "FALCONINSTRUCT"};

LLAMA_API t_finetune_type falcon_detect_finetune(falcon_context *ctx,
                                                 std::string model_path);
// Special tokens
LLAMA_API falcon_token falcon_token_bos();
LLAMA_API falcon_token falcon_token_eos();
LLAMA_API falcon_token falcon_token_nl();

// Sampling functions

/// @details Repetition penalty described in CTRL academic paper
/// https://arxiv.org/abs/1909.05858, with negative logit fix.
LLAMA_API void falcon_sample_repetition_penalty(
    struct falcon_context *ctx, falcon_token_data_array *candidates,
    const falcon_token *last_tokens, size_t last_tokens_size, float penalty);

/// @details Frequency and presence penalties described in OpenAI API
/// https://platform.openai.com/docs/api-reference/parameter-details.
LLAMA_API void falcon_sample_frequency_and_presence_penalties(
    struct falcon_context *ctx, falcon_token_data_array *candidates,
    const falcon_token *last_tokens, size_t last_tokens_size,
    float alpha_frequency, float alpha_presence);

/// @details Sorts candidate tokens by their logits in descending order and
/// calculate probabilities based on logits.
LLAMA_API void falcon_sample_softmax(struct falcon_context *ctx,
                                     falcon_token_data_array *candidates);
// logarithmic scaled softmax (just a log after softmax)
LLAMA_API void falcon_sample_log_softmax(struct falcon_context *ctx,
                                         falcon_token_data_array *candidates);

/// @details Top-K sampling described in academic paper "The Curious Case of
/// Neural Text Degeneration" https://arxiv.org/abs/1904.09751
LLAMA_API void falcon_sample_top_k(struct falcon_context *ctx,
                                   falcon_token_data_array *candidates, int k,
                                   size_t min_keep);

/// @details Nucleus sampling described in academic paper "The Curious Case of
/// Neural Text Degeneration" https://arxiv.org/abs/1904.09751
LLAMA_API void falcon_sample_top_p(struct falcon_context *ctx,
                                   falcon_token_data_array *candidates, float p,
                                   size_t min_keep);

/// @details Tail Free Sampling described in
/// https://www.trentonbricken.com/Tail-Free-Sampling/.
LLAMA_API void falcon_sample_tail_free(struct falcon_context *ctx,
                                       falcon_token_data_array *candidates,
                                       float z, size_t min_keep);

/// @details Locally Typical Sampling implementation described in the paper
/// https://arxiv.org/abs/2202.00666.
LLAMA_API void falcon_sample_typical(struct falcon_context *ctx,
                                     falcon_token_data_array *candidates,
                                     float p, size_t min_keep);
LLAMA_API void falcon_sample_temperature(struct falcon_context *ctx,
                                         falcon_token_data_array *candidates,
                                         float temp);

/// @details Mirostat 1.0 algorithm described in the paper
/// https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `falcon_token_data` containing the candidate
/// tokens, their probabilities (p), and log-odds (logit) for the current
/// position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve
/// for the generated text. A higher value corresponds to more surprising or
/// less predictable text, while a lower value corresponds to less surprising or
/// more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between
/// the target and observed surprisal of the sampled word. A larger learning
/// rate will cause `mu` to be updated more quickly, while a smaller learning
/// rate will result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This
/// is an arbitrary value that is used to calculate `s_hat`, which in turn helps
/// to calculate the value of `k`. In the paper, they use `m = 100`, but you can
/// experiment with different values to see how it affects the performance of
/// the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the
/// target cross-entropy (`2 * tau`) and is updated in the algorithm based on
/// the error between the target and observed surprisal.
LLAMA_API falcon_token falcon_sample_token_mirostat(
    struct falcon_context *ctx, falcon_token_data_array *candidates, float tau,
    float eta, int m, float *mu);

/// @details Mirostat 2.0 algorithm described in the paper
/// https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `falcon_token_data` containing the candidate
/// tokens, their probabilities (p), and log-odds (logit) for the current
/// position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve
/// for the generated text. A higher value corresponds to more surprising or
/// less predictable text, while a lower value corresponds to less surprising or
/// more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between
/// the target and observed surprisal of the sampled word. A larger learning
/// rate will cause `mu` to be updated more quickly, while a smaller learning
/// rate will result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the
/// target cross-entropy (`2 * tau`) and is updated in the algorithm based on
/// the error between the target and observed surprisal.
LLAMA_API falcon_token falcon_sample_token_mirostat_v2(
    struct falcon_context *ctx, falcon_token_data_array *candidates, float tau,
    float eta, float *mu);

/// @details Selects the token with the highest probability.
LLAMA_API falcon_token falcon_sample_token_greedy(
    struct falcon_context *ctx, falcon_token_data_array *candidates);

/// @details Randomly selects a token from the candidates based on their
/// probabilities.
LLAMA_API falcon_token falcon_sample_token(struct falcon_context *ctx,
                                           falcon_token_data_array *candidates);

// Performance information
LLAMA_API void falcon_print_timings(struct falcon_context *ctx);
LLAMA_API void falcon_reset_timings(struct falcon_context *ctx);

// Print system information
LLAMA_API const char *falcon_print_system_info(int n_threads, int n_cores);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_API_INTERNAL

#include <string>
#include <vector>
struct ggml_tensor;

std::vector<std::pair<std::string, struct ggml_tensor *>>
    &llama_internal_get_tensor_map(struct falcon_context *ctx);

#endif

#endif
