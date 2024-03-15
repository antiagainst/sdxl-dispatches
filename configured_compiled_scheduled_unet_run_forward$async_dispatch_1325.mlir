hal.executable public @run_forward$async_dispatch_1325 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_1325_generic_2x64x64x1280_i64xi64xi64xi64xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_1325_generic_2x64x64x1280_i64xi64xi64xi64xf16() {
        %c0_i64 = arith.constant 0 : i64
        %c2_i64 = arith.constant 2 : i64
        %c1280_i64 = arith.constant 1280 : i64
        %c32_i64 = arith.constant 32 : i64
        %c115897728 = arith.constant 115897728 : index
        %c6208 = arith.constant 6208 : index
        %c6272 = arith.constant 6272 : index
        %c16512 = arith.constant 16512 : index
        %c126383488 = arith.constant 126383488 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c115897728) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280x32x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c6208) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2xi64>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c6272) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xi64>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c16512) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64xi64>>
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c126383488) : !flow.dispatch.tensor<readwrite:tensor<2x66x66x1280xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 1280, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280x32x32xf32>> -> tensor<2x1280x32x32xf32>
        %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xi64>> -> tensor<2xi64>
        %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xi64>> -> tensor<1280xi64>
        %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:tensor<64xi64>> -> tensor<64xi64>
        %9 = tensor.empty() : tensor<2x64x64x1280xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7, %8, %8 : tensor<2xi64>, tensor<1280xi64>, tensor<64xi64>, tensor<64xi64>) outs(%9 : tensor<2x64x64x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 256]]>} {
        ^bb0(%in: i64, %in_0: i64, %in_1: i64, %in_2: i64, %out: f16):
          %11 = arith.cmpi slt, %in, %c0_i64 : i64
          %12 = arith.addi %in, %c2_i64 : i64
          %13 = arith.select %11, %12, %in : i64
          %14 = arith.index_cast %13 : i64 to index
          %15 = arith.cmpi slt, %in_0, %c0_i64 : i64
          %16 = arith.addi %in_0, %c1280_i64 : i64
          %17 = arith.select %15, %16, %in_0 : i64
          %18 = arith.index_cast %17 : i64 to index
          %19 = arith.cmpi slt, %in_1, %c0_i64 : i64
          %20 = arith.addi %in_1, %c32_i64 : i64
          %21 = arith.select %19, %20, %in_1 : i64
          %22 = arith.index_cast %21 : i64 to index
          %23 = arith.cmpi slt, %in_2, %c0_i64 : i64
          %24 = arith.addi %in_2, %c32_i64 : i64
          %25 = arith.select %23, %24, %in_2 : i64
          %26 = arith.index_cast %25 : i64 to index
          %extracted = tensor.extract %5[%14, %18, %22, %26] : tensor<2x1280x32x32xf32>
          %27 = arith.truncf %extracted : f32 to f16
          linalg.yield %27 : f16
        } -> tensor<2x64x64x1280xf16>
        flow.dispatch.tensor.store %10, %4, offsets = [0, 1, 1, 0], sizes = [2, 64, 64, 1280], strides = [1, 1, 1, 1] : tensor<2x64x64x1280xf16> -> !flow.dispatch.tensor<readwrite:tensor<2x66x66x1280xf16>>
        return
      }
    }
  }
}
