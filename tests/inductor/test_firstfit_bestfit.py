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

"""Tests for FirstFitLayoutSolver and BestFitLayoutSolver."""

from unittest import TestCase
from typing import TYPE_CHECKING

# Currently, it appears to be impossible to import torch_spyre without importing torch first.
import torch  # noqa: F401
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    BestFitLayoutSolver,
    FirstFitLayoutSolver,
    _assert_in_place_relationships,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

LARGE_SIZE = 512

if TYPE_CHECKING:
    BaseClass = TestCase
else:
    BaseClass = object


def buf(name: str, size: int, start: int, end: int) -> LifetimeBoundBuffer:
    return LifetimeBoundBuffer(name, size, start, end)


class LayoutSolverTests(BaseClass):
    """Behavioural tests shared by both solver subclasses."""

    solver_class: type[FirstFitLayoutSolver] = None  # type: ignore[assignment]

    def solve(self, buffers, size=LARGE_SIZE, alignment=1):
        return self.solver_class(size, alignment).plan_layout(buffers)

    def verify(self, buffers, expected, size=LARGE_SIZE, alignment=1):
        result = self.solve(buffers, size, alignment)
        self.assertEqual([b.address for b in result], expected)

    def test_empty_returns_empty_list(self):
        self.assertEqual(self.solve([]), [])

    def test_single_buffer_placed_at_zero(self):
        self.verify([buf("a", 10, 0, 5)], [0])

    def test_single_buffer_evicted_when_too_large(self):
        self.verify([buf("a", 11, 0, 5)], [None], size=10)

    def test_non_overlapping_lifetimes_reuse_address(self):
        # b1 ends at time 5 (exclusive); b2 starts at time 5 — they never coexist.
        self.verify([buf("b1", 20, 0, 5), buf("b2", 20, 5, 10)], [0, 0])

    def test_concurrent_buffers_packed_input_order(self):
        # Equal lifetimes: stable sort preserves input order, so a(10)@0, b(20)@10, c(30)@30.
        self.verify(
            [buf("a", 10, 0, 4), buf("b", 20, 0, 4), buf("c", 30, 0, 4)],
            [0, 10, 30],
            size=60,
        )

    def test_largest_buffer_evicted_when_full(self):
        # a(10)@0 and b(20)@10 consume 30 bytes; c(30) needs 30 but only 20 remain → evicted.
        self.verify(
            [buf("a", 10, 0, 4), buf("b", 20, 0, 4), buf("c", 30, 0, 4)],
            [0, 10, None],
            size=50,
        )

    def test_alignment_pads_between_buffers(self):
        # Two same-size concurrent buffers; the second is placed at the next
        # alignment boundary after the first.
        self.verify(
            [buf("a", 10, 0, 4), buf("b", 10, 0, 4)],
            [0, 128],
            alignment=128,
        )

    def test_alignment_can_cause_eviction(self):
        # a(13)@0 leaves a gap starting at 13; rounding up to alignment=10 gives
        # addr=20, but 20+12=32 > limit=30, so b is evicted.
        self.verify(
            [buf("a", 13, 0, 5), buf("b", 12, 0, 5)],
            [0, None],
            size=30,
            alignment=10,
        )


def _inplace_buf(name: str, size: int, start: int, end: int, parents: list[str]):
    return LifetimeBoundBuffer(name, size, start, end, in_place_parents=parents)


class InPlaceSolverTests(LayoutSolverTests):
    """In-place reuse tests shared by both solver subclasses."""

    def test_child_reuses_parent_address(self):
        # P ends at 5; C.start_time=4 == P.end_time - 1, so in-place is valid.
        # Without in-place, P's [0,20) would be subtracted and C would land at 20.
        p = LifetimeBoundBuffer("P", 20, 0, 5)
        c = _inplace_buf("C", 15, 4, 9, ["P"])
        result = self.solve([p, c])
        by_name = {b.name: b.address for b in result}
        self.assertEqual(by_name["P"], 0)
        self.assertEqual(by_name["C"], 0)

    def test_child_evicted_when_parent_evicted(self):
        # P is too large to fit; C declared as in-place child of P.
        # P gets evicted (address=None), so C also cannot in-place and
        # must fall back to normal placement.
        p = LifetimeBoundBuffer("P", 200, 0, 5)
        c = _inplace_buf("C", 15, 4, 9, ["P"])
        result = self.solve([p, c], size=100)
        by_name = {b.name: b.address for b in result}
        self.assertIsNone(by_name["P"])
        # C can still be placed independently (no overlap conflict with evicted P).
        self.assertEqual(by_name["C"], 0)

    def test_assert_rejects_wrong_end_time(self):
        p = LifetimeBoundBuffer("P", 20, 0, 5)
        c = _inplace_buf("C", 15, 3, 9, ["P"])  # start_time=3, need P.end_time==4
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])

    def test_assert_rejects_oversized_child(self):
        p = LifetimeBoundBuffer("P", 10, 0, 5)
        c = _inplace_buf("C", 15, 4, 9, ["P"])  # child larger than parent
        with self.assertRaises(AssertionError):
            _assert_in_place_relationships([p, c])


def _two_gap_buffers():
    """Buffers that leave two free gaps for x in a 120-byte scratchpad.

    Processing order by ascending lifetime: b_mid(2), b_left(4), b_right(5), x(5).
    b_right and x tie on lifetime; stable sort keeps b_right first.

    Placements: b_mid@0, b_left@40, b_right@70.
    b_mid lives [2,4) and x lives [4,9) — they do not overlap, so b_mid's
    address range (0,40) is not subtracted from x's gaps. After removing
    b_left(40,70) and b_right(70,100), x sees two gaps:
      (0,40)   waste = 30
      (100,120) waste = 10
    FirstFit picks (0,40) → addr=0; BestFit picks (100,120) → addr=100.
    """
    return [
        buf("b_mid", 40, 2, 4),
        buf("b_left", 30, 1, 5),
        buf("b_right", 30, 3, 8),
        buf("x", 10, 4, 9),
    ]


class TestFirstFitLayoutSolver(InPlaceSolverTests, TestCase):
    solver_class = FirstFitLayoutSolver

    def test_picks_first_gap_not_tightest(self):
        result = FirstFitLayoutSolver(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 0)


class TestBestFitLayoutSolver(InPlaceSolverTests, TestCase):
    solver_class = BestFitLayoutSolver

    def test_picks_tightest_gap(self):
        result = BestFitLayoutSolver(120, 1).plan_layout(_two_gap_buffers())
        x_addr = next(b.address for b in result if b.name == "x")
        self.assertEqual(x_addr, 100)


if __name__ == "__main__":
    import unittest

    unittest.main()
