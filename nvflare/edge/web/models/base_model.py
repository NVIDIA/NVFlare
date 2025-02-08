from urllib.parse import parse_qs, urlencode


class BaseModel(dict):
    """Dictionary based model"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def from_query_string(self, qs: str):
        params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(qs).items()}
        self.update(params)

    def to_query_string(self) -> str:
        return urlencode(self, doseq=True)
