from ctransformers import LLM, Config


class MockLLM(LLM):
    def __init__(self):
        self._config = Config()
        self._llm = None

    @property
    def config(self):
        return self._config

    def tokenize(self, prompt, **kwargs):
        self.tokens = prompt.split(" ")
        return range(len(self.tokens))

    def generate(self, tokens, **kwargs):
        return tokens

    def detokenize(self, tokens, decode=True):
        text = " " + self.tokens[tokens[0]]
        if not decode:
            text = text.encode()
        return text


class TestLLM:
    def test_stop(self):
        llm = MockLLM()
        prompt = "foo bar baz lorem ipsum\ndolor"
        expected = [
            ([], " foo bar baz lorem ipsum\ndolor"),
            (["dolor "], " foo bar baz lorem ipsum\ndolor"),
            (["ipsum "], " foo bar baz lorem ipsum\ndolor"),
            (["doloro"], " foo bar baz lorem ipsum\ndolor"),
            (["ipsumo"], " foo bar baz lorem ipsum\ndolor"),
            (["dolor"], " foo bar baz lorem ipsum\n"),
            (["ipsum"], " foo bar baz lorem "),
            (["olor"], " foo bar baz lorem ipsum\nd"),
            (["olo"], " foo bar baz lorem ipsum\nd"),
            (["psum"], " foo bar baz lorem i"),
            (["psu"], " foo bar baz lorem i"),
            (["z lor"], " foo bar ba"),
            (["rem", "or"], " foo bar baz l"),
            (["foo"], " "),
            (["f"], " "),
            ([" "], ""),
            (["\n"], " foo bar baz lorem ipsum"),
            (["m\nd"], " foo bar baz lorem ipsu"),
        ]
        for stop, response in expected:
            assert llm(prompt, stop=stop) == response
            if len(stop) == 1:
                assert llm(prompt, stop=stop[0]) == response
