from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_repo_id, HFValidationError

from ..llm import Config
from .llm import LLM


def get_path_type(path: str) -> Optional[str]:
    p = Path(path)
    if p.is_file():
        return "file"
    elif p.is_dir():
        return "dir"
    try:
        validate_repo_id(path)
        return "repo"
    except HFValidationError:
        pass


class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        model_path_or_repo_id: str,
        *,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ) -> LLM:
        """Loads the language model from a local file or remote repo.

        Args:
            model_path_or_repo_id: The path to a model file or directory or the
            name of a Hugging Face Hub model repo.
            local_files_only: Whether or not to only look at local files
            (i.e., do not try to download the model).
            revision: The specific model version to use. It can be a branch
            name, a tag name, or a commit id.

        Returns:
            `LLM` object.
        """
        config = Config()
        for k, v in kwargs.items():
            if not hasattr(config, k):
                raise TypeError(
                    f"'{k}' is an invalid keyword argument for from_pretrained()"
                )
            setattr(config, k, v)

        path_type = get_path_type(model_path_or_repo_id)
        if not path_type:
            raise ValueError(f"Model path '{model_path_or_repo_id}' doesn't exist.")

        model_path = None
        if path_type == "file":
            model_path = Path(model_path_or_repo_id).parent
        elif path_type == "dir":
            model_path = Path(model_path_or_repo_id)
        elif path_type == "repo":
            model_path = snapshot_download(
                repo_id=model_path_or_repo_id,
                local_files_only=local_files_only,
                revision=revision,
            )

        return LLM(model_path=model_path, config=config)
