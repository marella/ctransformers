import os
import pytest
import shutil
from ctransformers import AutoModelForCausalLM

@pytest.fixture
def temp_folder():
    # Setup: Create the .temp folder
    os.makedirs('.temp', exist_ok=False)
    
    # This will return control to the test function
    yield
    
    # Teardown: Remove the .temp folder after the test is done
    shutil.rmtree('.temp')

class TestModel:
    def test_generate(self, lib, temp_folder):
        llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", 
                                                   lib=lib,
                                                   cache_dir='.temp')
        assert os.path.exists('.temp'), "Temp file not created."
        assert os.path.isdir('.temp'), ".Temp file is not a folder"
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
