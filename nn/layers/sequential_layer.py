import pdb
from typing import Union, List, Tuple
from .layer import Layer
from .layer_using_layer import LayerUsingLayer


class SequentialLayer(LayerUsingLayer):
    def __init__(self, layers: Union[Tuple[Layer], List[Layer]], parent=None):
        super(SequentialLayer, self).__init__(parent)
        self.layers = layers
        parent = self.parent # this actually provides the true parent!
        for ll, layer in enumerate(self.layers):
            setattr(self, str(ll), layer)
            layer.parent = parent
            if isinstance(layer, LayerUsingLayer): # resblock is getting into this, maybe final layer causing the parent problem
                parent = layer.final_layer
            else:
                parent = layer

    def forward(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def __getitem__(self, item):
        return self.layers[item]

    @property
    def final_layer(self):
        final_layer = self.layers[-1]
        if isinstance(final_layer, LayerUsingLayer):
            return final_layer.final_layer
        return final_layer
