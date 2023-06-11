from ctransformers import AutoModelForCausalLM


class TestModel:
    def test_generate(self, lib):
        llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", lib=lib)
        response = llm("AI is going to", seed=5, max_new_tokens=3)
        assert response == " be a big"

        token = llm.sample()
        logits = llm.logits
        value = logits[token]
        logits[token] -= 1
        assert logits[token] == llm.logits[token] == value - 1
        llm.logits[token] *= 2
        assert logits[token] == llm.logits[token] == (value - 1) * 2

        assert llm.eos_token_id == 50256
        assert llm.vocab_size == 50257
        assert llm.context_length == 1024
