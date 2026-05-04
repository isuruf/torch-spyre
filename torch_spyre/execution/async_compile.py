# Copyright 2025 The Torch-Spyre Authors.
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

from collections.abc import Sequence
from typing import Any
import subprocess
import torch

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre.execution.compile_cache import get_spyre_cache
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        cache = get_spyre_cache()
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        op_specs = [s for s in specs if isinstance(s, (OpSpec, LoopSpec))]

        output_dir, cache_found = cache.try_load(op_specs)
        if not cache_found:
            # Generate SDSC Bundle from OpSpecs
            generate_bundle(kernel_name, output_dir, op_specs)
            # Invoke backend compiler of SDSC Bundle
            with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
                subprocess.run(["dxp_standalone", "-d", output_dir], check=True)
            convert_artifacts(output_dir)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
