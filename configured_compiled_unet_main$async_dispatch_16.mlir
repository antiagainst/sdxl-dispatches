hal.executable public @main$async_dispatch_16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_16_matmul_transpose_b_2x320x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [320 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_16_matmul_transpose_b_2x320x1280_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c268544 = arith.constant 268544 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c268544) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x320xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>> -> tensor<2x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [320, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<320x1280xf16>> -> tensor<320x1280xf16>
        %5 = tensor.empty() : tensor<2x320xf32>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1280]]>} ins(%cst : f32) outs(%5 : tensor<2x320xf32>) -> tensor<2x320xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<2x1280xf16>, tensor<320x1280xf16>) outs(%6 : tensor<2x320xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1280]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<2x320xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 320], strides = [1, 1] : tensor<2x320xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x320xf32>>
        return
      }
    }
  }
}
