hal.executable public @main$async_dispatch_15 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_15_matmul_transpose_b_2x1280x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [320 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_15_matmul_transpose_b_2x1280x1280_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+00 : f16
        %c263424 = arith.constant 263424 : index
        %c273664 = arith.constant 273664 : index
        %c0 = arith.constant 0 : index
        %c5120 = arith.constant 5120 : index
        %c15360 = arith.constant 15360 : index
        %c268544 = arith.constant 268544 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c263424) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c5120) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c273664) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>>
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c15360) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c268544) : !flow.dispatch.tensor<writeonly:tensor<2x1280xf16>>
        %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf16>> -> tensor<2x1280xf16>
        %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
        %8 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>> -> tensor<2x1280xf32>
        %10 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %11 = tensor.empty() : tensor<2x1280xf16>
        %12 = tensor.empty() : tensor<2x1280xf32>
        %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1280]]>} ins(%cst : f32) outs(%12 : tensor<2x1280xf32>) -> tensor<2x1280xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<2x1280xf16>, tensor<1280x1280xf16>) outs(%13 : tensor<2x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1280]]>} {
        ^bb0(%in: f16, %in_1: f16, %out: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_1 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %out, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<2x1280xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %8, %9, %10 : tensor<2x1280xf32>, tensor<1280xf32>, tensor<2x1280xf32>, tensor<1280xf32>) outs(%11 : tensor<2x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1280]]>} {
        ^bb0(%in: f32, %in_1: f32, %in_2: f32, %in_3: f32, %out: f16):
          %16 = arith.addf %in, %in_1 : f32
          %17 = arith.addf %in_2, %in_3 : f32
          %18 = arith.truncf %17 : f32 to f16
          %19 = arith.truncf %16 : f32 to f16
          %20 = arith.addf %19, %18 : f16
          %21 = arith.negf %20 : f16
          %22 = math.exp %21 : f16
          %23 = arith.addf %22, %cst_0 : f16
          %24 = arith.divf %cst_0, %23 : f16
          %25 = arith.mulf %24, %20 : f16
          linalg.yield %25 : f16
        } -> tensor<2x1280xf16>
        flow.dispatch.tensor.store %15, %5, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : tensor<2x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1280xf16>>
        return
      }
    }
  }
}
