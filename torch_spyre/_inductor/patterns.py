# Copyright 2026 The Torch-Spyre Authors.
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


import torch
from torch._inductor.fx_passes.joint_graph import (
    PatternMatcherPass,
    register_graph_pattern,
    Arg,
    CallFunction,
    Match,
)

aten = torch.ops.aten
spyre_pass_pattern_dict = PatternMatcherPass()


def silu_patterns():
    x = Arg()
    one = Arg()
    neg = CallFunction(aten.neg.default, x)
    exp = CallFunction(aten.exp.default, neg)
    adds = [
        CallFunction(aten.add.Tensor, exp, one),
        CallFunction(aten.add.Tensor, one, exp),
        CallFunction(aten.add.Scalar, exp, one),
    ]
    for add in adds:
        yield CallFunction(aten.div.Tensor, x, add)


def silu_check_one(match):
    one = match.args[1]
    return isinstance(one, int) and one == 1


def silu_pattern_replace(match: Match, x, one):
    # We replae silu patterns with spyre.silu custom op
    # that directly lowers to the silu OpFunc and avoids
    # the need to have 4 OpFuncs
    match.replace_by_example(torch.ops.spyre.silu, [x])


for silu_pattern in silu_patterns():
    register_graph_pattern(
        silu_pattern,
        pass_dict=spyre_pass_pattern_dict,
        extra_check=silu_check_one,
    )(silu_pattern_replace)
