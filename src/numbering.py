from abc import ABC, abstractmethod
from dataclasses import dataclass


class Numbering(ABC):
    @abstractmethod
    def indices(self) -> list[dict[str, int]]:
        pass


@dataclass
class ParameterNumbering(Numbering):
    scan_order: str  # "params first" or "reps first"
    n_reps: int  # number of repetitions of the same parameter value
    n_parameters: int  # number of parameters to scan
    images_per_cycle: int  # number of images per cycle
    parameters: list[float] = None  # list of parameter values
    parameter_name: str = None  # name of the parameter

    def indices(self) -> list[dict[str, int]]:
        if self.scan_order == "params first":
            for p in range(self.n_parameters):
                for r in range(self.n_reps):
                    for i in range(self.images_per_cycle):
                        yield {
                            "parameter_index": p,
                            "repetition_index": r,
                            "image_index": i,
                            "image_type": 1 if i == self.images_per_cycle - 1 else 0,
                        }
        elif self.scan_order == "reps first":
            for r in range(self.n_reps):
                for p in range(self.n_parameters):
                    for i in range(self.images_per_cycle):
                        yield {
                            "parameter_index": p,
                            "repetition_index": r,
                            "image_index": i,
                            "image_type": 1 if i == self.images_per_cycle - 1 else 0,
                        }
        else:
            raise ValueError("Invalid scan order")


@dataclass
class DynamicImageNumbering(Numbering):
    n_max: int  # maximum value for $n$
    n_reps: int  # number of repetitions for each value of $n$

    def indices(self) -> list[dict[str, int]]:
        for r in range(self.n_reps):
            for n in range(self.n_max):
                yield {"n": n, "image_index": 0, "repetition_index": r}
                for image_i in range(n):
                    yield {"n": n, "image_index": image_i+1, "repetition_index": r}
                yield {"n": n, "image_index": n + 1, "repetition_index": r}

@dataclass
class ClassicalNumbering(Numbering):
    n_reps: int

    def indices(self) -> list[dict[str, int]]:
        for r in range(self.n_reps):
                yield {"image_index": 0, "repetition_index": r}
                yield {"image_index": 1, "repetition_index": r}