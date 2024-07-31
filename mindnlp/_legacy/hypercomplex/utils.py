# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utils"""
from typing import Tuple, Union
import mindspore
from mindspore import ops as P


to_complex = P.Complex()
get_real = P.Real()
get_imag = P.Imag()


def get_x_and_y(tensor):
    """get real and image data from the tensor"""
    if tensor.dtype == mindspore.complex64:
        return get_real(tensor), get_imag(tensor)
    return P.unstack(tensor, 0)


def to_2channel(real, imag, dtype=None):
    """combine real and image data to a complex number"""
    if dtype is not None and dtype == mindspore.complex64:
        return to_complex(real, imag)
    if dtype is not None and (dtype != real.dtype or dtype != imag.dtype):
        raise ValueError("dtype must match with data type of the input tensors, but got: "
                         f"dtype={dtype}, real.dtype={real.dtype}, imag.dtype={imag.dtype}")
    real = P.expand_dims(real, 0)
    imag = P.expand_dims(imag, 0)
    return P.concat((real, imag), 0)


_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]
_size_any_t = Union[int, Tuple[int, ...]]
