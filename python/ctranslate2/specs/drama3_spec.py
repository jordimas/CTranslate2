"""Drama3 checkpoint layout for standard CTranslate2 conversion."""

from ctranslate2.specs import model_spec


class _ParameterSpec(model_spec.LayerSpec):
    def __init__(self, parameters):
        for name, value in parameters.items():
            setattr(
                self, name, _ParameterSpec(value) if isinstance(value, dict) else value
            )


class Drama3Spec(model_spec.ModelSpec):
    def __init__(self, variables, config=None, model_name="Drama3Spec"):
        super().__init__()
        self._model_name = model_name
        self._config = model_spec.ModelConfig(**config) if config is not None else None
        parameters = {}
        for path, value in variables.items():
            parts = path.split("/")
            if any(not part or part.startswith("_") for part in parts):
                raise ValueError("Invalid Drama3 parameter path: %s" % path)
            parent = parameters
            for part in parts[:-1]:
                parent = parent.setdefault(part, {})
            parent[parts[-1]] = value
        for name, value in parameters.items():
            setattr(
                self, name, _ParameterSpec(value) if isinstance(value, dict) else value
            )

    @property
    def name(self):
        return self._model_name
