# Copyright 2021 NVIDIA Corporation
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

from __future__ import absolute_import

from .add_tests import (
    broadcast,
    complex_data,
    inplace_broadcast,
    inplace_normal,
    normal,
    operator_inplace_broadcast,
    operator_inplace_normal,
    operator_normal,
    scalar,
)


def test():
    broadcast.test()
    inplace_broadcast.test()
    inplace_normal.test()
    normal.test()
    operator_normal.test()
    operator_inplace_broadcast.test()
    operator_inplace_normal.test()
    operator_normal.test()
    scalar.test()
    complex_data.test()


if __name__ == "__main__":
    test()
