hal.executable public @main$async_dispatch_57 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_57_matmul_transpose_b_8192x640x2560_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 5, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_57_matmul_transpose_b_8192x640x2560_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x2560xf16>>
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640x2560xf16>>
        %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %11 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %12 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        %13 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [8192, 2560], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x2560xf16>> -> tensor<8192x2560xf16>
        %14 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [640, 2560], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<640x2560xf16>> -> tensor<640x2560xf16>
        %15 = flow.dispatch.tensor.load %10, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %17 = tensor.empty() : tensor<8192x640xf16>
        %18 = tensor.empty() : tensor<8192x640xf32>
        %19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>} ins(%cst : f32) outs(%18 : tensor<8192x640xf32>) -> tensor<8192x640xf32>
        %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<8192x2560xf16>, tensor<640x2560xf16>) outs(%19 : tensor<8192x640xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %22 = arith.extf %in : f16 to f32
          %23 = arith.extf %in_0 : f16 to f32
          %24 = arith.mulf %22, %23 : f32
          %25 = arith.addf %out, %24 : f32
          linalg.yield %25 : f32
        } -> tensor<8192x640xf32>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %15, %16 : tensor<8192x640xf32>, tensor<640xf32>, tensor<8192x640xf16>) outs(%17 : tensor<8192x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f16):
          %22 = arith.addf %in, %in_0 : f32
          %23 = arith.truncf %22 : f32 to f16
          %24 = arith.addf %23, %in_1 : f16
          linalg.yield %24 : f16
        } -> tensor<8192x640xf16>
        flow.dispatch.tensor.store %21, %12, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : tensor<8192x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        return
      }
    }
  }
}
