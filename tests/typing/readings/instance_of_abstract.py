"""The instances of an abstract class, through `instance_of`."""

import abc

from typing_extensions import reveal_type

from valgebra import instance_of


class Shape(abc.ABC):
    @abc.abstractmethod
    def area(self) -> float: ...


reveal_type(instance_of(Shape))
