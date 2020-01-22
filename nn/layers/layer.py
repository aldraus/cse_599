from collections import OrderedDict
from typing import Callable, List, Tuple, Iterable, Dict, Optional, Union

import numpy as np

from nn import Parameter


def normals_initializer(parameter: Parameter):
    if len(parameter.data.squeeze().shape) == 1:
        # Zero bias initializer
        parameter.data = 0
    else:
        # Normal distribution
        parameter.data = np.random.normal(0, 0.1, parameter.data.shape)


class Layer(object):
    def __init__(self, parent: Optional[Union["Layer", Tuple["Layer", ...]]] = None):
        assert (
            parent is None or isinstance(parent, Layer) or isinstance(parent, Tuple)
        ), "Parents must be a Layer, a list/tuple of Layers, or None"

        self._parent = self._fix_parent(parent)

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def _fix_parent(parent):
        if parent is not None:
            if isinstance(parent, Layer):
                if hasattr(parent, "final_layer"):
                    parent = parent.final_layer
            else:
                final_layers = []
                for pp in parent:
                    if hasattr(pp, "final_layer"):
                        pp = pp.final_layer
                    final_layers.append(pp)
                parent = tuple(final_layers)
        return parent

    @property
    def parent(self) -> Optional[Union[Tuple["Layer", ...], "Layer"]]:
        self._parent = self._fix_parent(self._parent)
        return self._parent

    @parent.setter
    def parent(self, val: Union[Tuple["Layer", ...], "Layer"]):
        self.set_parent(val)

    def set_parent(self, val):
        self._parent = val

    @property
    def parents(self) -> Optional[Tuple["Layer", ...]]:
        if self.parent is None:
            return None
        if isinstance(self.parent, Tuple):
            return self.parent
        else:
            return (self.parent,)

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def backward(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError

    def vars(self):
        for obj, val in vars(self).items():
            if obj == "_parent":
                continue
            yield (obj, val)

    def state_dict(self, prefix="") -> Dict[str, Parameter]:
        state_dict = OrderedDict()
        for obj, val in self.vars():
            if isinstance(val, Layer):
                state_dict.update(val.state_dict(obj + "."))
            elif isinstance(val, Parameter):
                state_dict[prefix + obj] = val
        return state_dict

    def parameters(self) -> Iterable[Parameter]:
        return self.state_dict().values()

    def children(self) -> Iterable["Layer"]:
        for obj, val in self.vars():
            if isinstance(val, Layer):
                yield val

    def own_parameters(self) -> Iterable[Parameter]:
        params = []
        for obj, val in self.vars():
            if isinstance(val, Parameter):
                params.append(val)
        return params

    def share_parameters(self, other: "Layer"):
        assert type(self) == type(other)
        self_vars = sorted(self.vars())
        other_vars = sorted(other.vars())
        for ((self_obj, self_val), (other_obj, other_val)) in zip(self_vars, other_vars):
            if isinstance(self_val, Layer):
                self_val.share_parameters(other_val)
            elif isinstance(self_val, Parameter):
                assert self_val.data.shape == other_val.data.shape
                setattr(self, self_obj, other_val)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is not None:
            for param in self.own_parameters():
                initializer(param)
        for val in self.children():
            val.initialize()

    def selfstr(self) -> str:
        """
        Overload this function to print the specifics for this layer.
        See LinearLayer for an example.
        """
        return ""

    def _total_str(self, depth=0) -> List[str]:
        from nn.layers.dummy_layer import DummyLayer
        total_str_arr = []
        for obj, val in self.vars():
            if isinstance(val, Layer):
                if isinstance(val, DummyLayer):
                    total_str_arr.extend(val._total_str(depth))
                else:
                    total_str_arr.append(("(" + obj + "): " + val.name + ": " + val.selfstr(), depth))
                    # total_str_arr.append((val.selfstr(), depth + 1))
                    total_str_arr.extend(val._total_str(depth + 1))
        return total_str_arr

    def __str__(self):
        total_str_arr = list()
        total_str_arr.append((self.name + ": " + self.selfstr(), 0))
        # total_str_arr.append((self.selfstr(), 1))
        total_str_arr.extend(self._total_str(1))
        filtered_strs = []
        for val in total_str_arr:
            if val is None or len(val) == 0:
                continue
            string, depth = val
            if string is None or len(string) == 0:
                continue
            filtered_strs.append((string, depth))
        strs = [" " * (depth * 4) + string for string, depth in filtered_strs]
        return "\n".join(strs)

    __repr__ = __str__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
