# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Time series distributional output classes and utilities.
"""
from typing import Callable, Dict, Optional, Tuple
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.nn.probability.distribution import (
    Distribution,
    Normal,
    StudentT,
    TransformedDistribution,
    # AffineTransform,
    # Independent,
    # NegativeBinomial,
)
from mindspore.nn.probability.bijector import ScalarAffine as AffineTransform
import numpy as np

class AffineTransformed(TransformedDistribution):
    '''
    # todo 
    '''

    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc
        super().__init__(AffineTransform(shift=self.loc, scale=self.scale), base_distribution)

    def _set_attr_for_tensor(self, name, value):
        object.__setattr__(self, name, value)

    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


class ParameterProjection(nn.Cell):
    """
    # todo
    """
    def __init__(
        self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[mindspore.Tensor]], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.CellList([nn.Dense(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def construct(self, x: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Cell):
    """
    #todo
    """
    def __init__(self, function):
        super().__init__()
        self.function = function

    def construct(self, x, *args):
        return self.function(x, *args)


class DistributionOutput:
    """
    # todo
    """
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        #if self.dim == 1:
        return self.distribution_class(*distr_args)
        #return Independent(self.distribution_class(*distr_args), 1)

    def distribution(
        self,
        distr_args,
        loc: Optional[mindspore.Tensor] = None,
        scale: Optional[mindspore.Tensor] = None,
    ) -> Distribution:
        r"""
        # todo
        """
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    def get_parameter_projection(self, in_features: int) -> nn.Cell:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return ParameterProjection(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: mindspore.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    @staticmethod
    def squareplus(x: mindspore.Tensor) -> mindspore.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + ops.sqrt(ops.square(x) + 4.0)) / 2.0


class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """

    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: mindspore.Tensor, loc: mindspore.Tensor, scale: mindspore.Tensor):
        scale = cls.squareplus(scale).clamp(
            mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(scale.dtype)).eps))
        df = 2.0 + cls.squareplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """

    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: mindspore.Tensor, scale: mindspore.Tensor):
        scale = cls.squareplus(scale).clamp_min(np.finfo(mindspore.dtype_to_nptype(scale.dtype)).eps)
        return loc.squeeze(-1), scale.squeeze(-1)
