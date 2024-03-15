hal.executable public @main$async_dispatch_10 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_10_matmul_transpose_b_2x1280x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>, workgroup_size = [32 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_10_matmul_transpose_b_2x1280x1280_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %c6400 = arith.constant 6400 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c6400) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%1) : !flow.dispatch.tensor<writeonly:tensor<2x1280xf32>>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>> -> tensor<2x1280xf16>
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
        %7 = tensor.empty() : tensor<2x1280xf32>
        %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 8]]>} ins(%cst : f32) outs(%7 : tensor<2x1280xf32>) -> tensor<2x1280xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<2x1280xf16>, tensor<1280x1280xf16>) outs(%8 : tensor<2x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 8]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %10 = arith.extf %in : f16 to f32
          %11 = arith.extf %in_0 : f16 to f32
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %out, %12 : f32
          linalg.yield %13 : f32
        } -> tensor<2x1280xf32>
        flow.dispatch.tensor.store %9, %4, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : tensor<2x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x1280xf32>>
        return
      }
    }
  }
}
