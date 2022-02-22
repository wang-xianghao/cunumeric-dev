/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/index/repeat.h"
#include "cunumeric/index/repeat_template.inl"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(const AccessorWO<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                int64_t repeats,
                const int32_t axis,
                const Rect<DIM> rect,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto p            = pitches.unflatten(idx, rect.lo);
  int64_t input_idx = p[axis] / repeats;
  auto in_p         = p;
  in_p[axis]        = input_idx;
  out[p]            = in[in_p];
}

template <typename VAL, int DIM>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  repeat_kernel(const AccessorWO<VAL, DIM> out,
                const AccessorRO<VAL, DIM> in,
                const AccessorRO<int64_t, 1> repeats,
                const int32_t axis,
                const Rect<DIM> rect,
                const Pitches<DIM - 1> pitches,
                const int volume)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  int64_t total_repeats = repeats[0];
  int64_t input_idx     = 0;
  auto p                = pitches.unflatten(idx, rect.lo);
  while (total_repeats <= p[axis]) {
    input_idx++;
    total_repeats += repeats[input_idx];
  }
  auto in_p  = p;
  in_p[axis] = input_idx;
  out[p]     = in[in_p];
}

template <LegateTypeCode CODE, int DIM>
struct RepeatImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const int64_t repeats,
                  const int32_t axis,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    repeat_kernel<VAL, DIM>
      <<<blocks, THREADS_PER_BLOCK>>>(out, in, repeats, axis, rect, pitches, volume);
  }

  void operator()(const AccessorWO<VAL, DIM>& out,
                  const AccessorRO<VAL, DIM>& in,
                  const AccessorRO<int64_t, 1>& repeats,
                  const int32_t axis,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  size_t repeats_size) const
  {
    const size_t volume = rect.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    repeat_kernel<VAL, DIM>
      <<<blocks, THREADS_PER_BLOCK>>>(out, in, repeats, axis, rect, pitches, volume);
  }
};

/*static*/ void RepeatTask::gpu_variant(TaskContext& context)
{
  repeat_template<VariantKind::GPU>(context);
}
}  // namespace cunumeric
