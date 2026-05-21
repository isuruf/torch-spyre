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

import math
from typing import Optional

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)


def round_up_to_alignment(arg: int, alignment: int) -> int:
    return ((arg + alignment - 1) // alignment) * alignment


class FirstFitLayoutSolver(MemoryPlanSolver):
    """Allocates buffers greedily, largest-first, placing each in the first gap that fits.

    Buffers are sorted by descending size (ties broken by start_time) and placed one at a time. For
    each buffer, the set of address gaps that are free during its lifetime is tracked; the buffer is
    placed at the start of the first gap large enough to hold it (rounded up to alignment). Buffers
    that cannot fit within self.limit are evicted (address=None).
    """

    def _all_minus(
        self,
        intervals: list[tuple[int, int]],
        interval: tuple[int, int],
        minimum_size: int,
    ) -> list[tuple[int, int]]:
        """Return intervals with interval subtracted, dropping remainders < minimum_size."""
        result = []
        for a, b in intervals:
            if a < interval[0]:
                if b < interval[0]:
                    if b - a >= minimum_size:
                        result.append((a, b))
                else:
                    if interval[0] - a >= minimum_size:
                        result.append((a, interval[0]))
            if b > interval[1]:
                if a > interval[1]:
                    if b - a >= minimum_size:
                        result.append((a, b))
                else:
                    if b - interval[1] >= minimum_size:
                        result.append((interval[1], b))
        return result

    def _pick_address(
        self, large_gaps: list[tuple[int, int]], size: int
    ) -> Optional[int]:
        """Return the aligned start address in the first fitting gap, or None."""
        for gap in large_gaps:
            addr = round_up_to_alignment(gap[0], self.alignment)
            if addr + size <= gap[1]:
                return addr
        return None

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )

        # Omit buffers that are used in only one op.
        buffers_filtered = [
            buffer for buffer in buffers if buffer.end_time > buffer.start_time + 1
        ]
        # Prefer buffers that live only briefly. (Ideally, we would sort by the life time divided by
        # the number of uses, but we don't currently have access to that.)
        buffers_sorted = sorted(
            buffers_filtered, key=lambda buffer: buffer.end_time - buffer.start_time
        )

        for i, buffer in enumerate(buffers_sorted):
            large_gaps: list[tuple[int, int]] = [(0, self.limit)]

            for other_buffer in buffers_sorted[:i]:
                other_addr = other_buffer.address
                if other_addr is None:
                    continue
                # end_time is exclusive: buffer is alive in [start_time, end_time).
                if not (
                    other_buffer.start_time < buffer.end_time
                    and buffer.start_time < other_buffer.end_time
                ):
                    continue
                large_gaps = self._all_minus(
                    large_gaps,
                    (other_addr, other_addr + other_buffer.size),
                    buffer.size,
                )

            buffer.address = self._pick_address(large_gaps, buffer.size)

        return buffers


class BestFitLayoutSolver(FirstFitLayoutSolver):
    """Like FirstFitLayoutSolver but places each buffer in the tightest fitting gap.

    Inherits all logic from FirstFitLayoutSolver; only the gap-selection policy
    differs: instead of picking the first gap large enough to hold the buffer,
    this picks the gap that minimises leftover space after placement.
    """

    def _pick_address(
        self, large_gaps: list[tuple[int, int]], size: int
    ) -> Optional[int]:
        """Return the aligned start address in the tightest fitting gap, or None."""
        best_addr: Optional[int] = None
        best_waste: int = 0
        for gap in large_gaps:
            addr = math.ceil(gap[0] / self.alignment) * self.alignment
            if addr + size <= gap[1]:
                waste = gap[1] - addr - size
                if best_addr is None or waste < best_waste:
                    best_addr = addr
                    best_waste = waste
        return best_addr
