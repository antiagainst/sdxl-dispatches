hal.executable public @run_forward$async_dispatch_929 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_929_matmul_transpose_b_2048x1280x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 1, subgroup_m_tile_count = 2, subgroup_n_tile_count = 5, subgroup_k_tile_count = 4>}>, workgroup_size = [64 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_929_matmul_transpose_b_2048x1280x1280_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
        %9 = tensor.empty() : tensor<2048x1280xf32>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 80, 64]]>} ins(%cst : f32) outs(%9 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %8 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%10 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 80, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %12 = arith.extf %in : f16 to f32
          %13 = arith.extf %in_0 : f16 to f32
          %14 = arith.mulf %12, %13 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<2048x1280xf32>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        return
      }
    }
  }
}
