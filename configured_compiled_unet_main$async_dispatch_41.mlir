hal.executable public @main$async_dispatch_41 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640x640xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [640, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<640x640xf16>> -> tensor<640x640xf16>
        %10 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %11 = tensor.empty() : tensor<8192x640xf16>
        %12 = tensor.empty() : tensor<8192x640xf32>
        %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%12 : tensor<8192x640xf32>) -> tensor<8192x640xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8, %9 : tensor<8192x640xf16>, tensor<640x640xf16>) outs(%13 : tensor<8192x640xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_0 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %out, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<8192x640xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %10 : tensor<8192x640xf32>, tensor<640xf16>) outs(%11 : tensor<8192x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
        ^bb0(%in: f32, %in_0: f16, %out: f16):
          %16 = arith.truncf %in : f32 to f16
          %17 = arith.addf %16, %in_0 : f16
          linalg.yield %17 : f16
        } -> tensor<8192x640xf16>
        flow.dispatch.tensor.store %15, %7, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : tensor<8192x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        return
      }
    }
  }
}
