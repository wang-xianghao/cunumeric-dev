# Copyright 2021-2022 NVIDIA Corporation
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
#

import pytest

import cunumeric as np
from numpy.exceptions import AxisError

def test_functionality():
    s = (2, 3, 4, 5)
    a = np.empty(s)
    for axis in range(-5, 4):
        b = np.expand_dims(a, axis)
        assert b.shape[axis] == 1
        assert np.squeeze(b).shape == s

def test_axis_tuple():
    a = np.empty((3, 3, 3))
    assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
    assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
    assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
    assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

def test_axis_out_of_range():
    s = (2, 3, 4, 5)
    a = np.empty(s)
    pytest.raises(AxisError, np.expand_dims, a, -6)
    pytest.raises(AxisError, np.expand_dims, a, 5)

    a = np.empty((3, 3, 3))
    pytest.raises(AxisError, np.expand_dims, a, (0, -6))
    pytest.raises(AxisError, np.expand_dims, a, (0, 5))

def test_repeated_axis():
    a = np.empty((3, 3, 3))
    pytest.raises(ValueError, np.expand_dims, a, axis=(1, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
