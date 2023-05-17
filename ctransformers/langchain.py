try:
    from langchain.llms.base import LLM
except ImportError:
    raise ImportError(
        'To use the ctransformers.langchain module, please install the '
        'langchain package: pip install langchain')

from typing import Any, Dict, Optional, Sequence

from pydantic import root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun

from . import AutoModelForCausalLM


class CTransformers(LLM):
    client: Any  #: :meta private:
    model: str
    model_type: Optional[str] = None
    model_file: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    lib: Optional[Any] = None

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'model_type': self.model_type,
            'model_file': self.model_file,
            'config': self.config,
        }

    @property
    def _llm_type(self) -> str:
        return 'ctransformers'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        config = values['config'] or {}
        values['client'] = AutoModelForCausalLM.from_pretrained(
            values['model'],
            model_type=values['model_type'],
            model_file=values['model_file'],
            lib=values['lib'],
            **config,
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        text = []
        for chunk in self.client._stream(prompt, stop=stop):
            text.append(chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk, verbose=self.verbose)
        return ''.join(text)
