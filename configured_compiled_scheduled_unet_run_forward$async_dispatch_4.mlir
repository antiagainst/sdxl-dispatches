hal.executable public @run_forward$async_dispatch_4 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_4_generic_128x12_f16xf32xf32xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_4_generic_128x12_f16xf32xf32xf32() {
        %c0 = arith.constant 0 : index
        %c17024 = arith.constant 17024 : index
        %c268608 = arith.constant 268608 : index
        %c274752 = arith.constant 274752 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c17024) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c268608) : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c274752) : !flow.dispatch.tensor<writeonly:tensor<128x12xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xf16>> -> tensor<12xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
        %6 = tensor.empty() : tensor<12x128xf32>
        %7 = tensor.empty() : tensor<128x12xf32>
        %8:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : tensor<12xf16>, tensor<128xf32>) outs(%6, %7 : tensor<12x128xf32>, tensor<128x12xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128]]>} {
        ^bb0(%in: f16, %in_0: f32, %out: f32, %out_1: f32):
          %9 = arith.extf %in : f16 to f32
          %10 = arith.mulf %9, %in_0 : f32
          %11 = math.sin %10 : f32
          linalg.yield %10, %11 : f32, f32
        } -> (tensor<12x128xf32>, tensor<128x12xf32>)
        flow.dispatch.tensor.store %8#0, %2, offsets = [0, 0], sizes = [12, 128], strides = [1, 1] : tensor<12x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        flow.dispatch.tensor.store %8#1, %3, offsets = [0, 0], sizes = [128, 12], strides = [1, 1] : tensor<128x12xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x12xf32>>
        return
      }
    }
  }
}
