# https://github.com/tatp22/multidim-positional-encoding
# pip install positional-encodings[pytorch]
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer

def get_positional_encoding_1d(dim):
    return Summer(PositionalEncoding1D(dim))

def get_positional_encoding_2d(dim):
    return Summer(PositionalEncoding2D(dim))
