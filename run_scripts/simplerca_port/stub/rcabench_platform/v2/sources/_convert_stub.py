from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True, slots=True)
class Label:
    level: str
    name: str


class DatapackLoader:
    pass


class DatasetLoader:
    pass
