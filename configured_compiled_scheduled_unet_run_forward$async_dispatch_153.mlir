hal.executable public @run_forward$async_dispatch_153 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_153_generic_2048x1280_f16xf32xf32xf16xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [320 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_153_generic_2048x1280_f16xf32xf32xf16xf16xf16() {
        %cst = arith.constant 9.99999974E-6 : f32
        %cst_0 = arith.constant 1.280000e+03 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %c32832 = arith.constant 32832 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c32832) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048xf32>>
        %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
        %9 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %10 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [2048], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
        %11 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf16>> -> tensor<1280xf16>
        %12 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf16>> -> tensor<1280xf16>
        %13 = tensor.empty() : tensor<2048x1280xf16>
        %14 = tensor.empty() : tensor<2048xf32>
        %15 = tensor.empty() : tensor<2048x1280xf32>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2048x1280xf16>) outs(%15 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} {
        ^bb0(%in: f16, %out: f32):
          %22 = arith.extf %in : f16 to f32
          linalg.yield %22 : f32
        } -> tensor<2048x1280xf32>
        %17 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} ins(%cst_1 : f32) outs(%10 : tensor<2048xf32>) -> tensor<2048xf32>
        %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%16 : tensor<2048x1280xf32>) outs(%17 : tensor<2048xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} {
        ^bb0(%in: f32, %out: f32):
          %22 = arith.addf %in, %out : f32
          linalg.yield %22 : f32
        } -> tensor<2048xf32>
        %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<2048xf32>) outs(%14 : tensor<2048xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} {
        ^bb0(%in: f32, %out: f32):
          %22 = arith.divf %in, %cst_0 : f32
          linalg.yield %22 : f32
        } -> tensor<2048xf32>
        %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%16, %19 : tensor<2048x1280xf32>, tensor<2048xf32>) outs(%17 : tensor<2048xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %22 = arith.subf %in, %in_2 : f32
          %23 = arith.mulf %22, %22 : f32
          %24 = arith.addf %23, %out : f32
          linalg.yield %24 : f32
        } -> tensor<2048xf32>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %19, %20, %11, %12 : tensor<2048x1280xf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<1280xf16>, tensor<1280xf16>) outs(%13 : tensor<2048x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 1280]]>} {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f16, %in_5: f16, %out: f16):
          %22 = arith.divf %in_3, %cst_0 : f32
          %23 = arith.addf %22, %cst : f32
          %24 = math.rsqrt %23 : f32
          %25 = arith.extf %in : f16 to f32
          %26 = arith.subf %25, %in_2 : f32
          %27 = arith.mulf %26, %24 : f32
          %28 = arith.extf %in_4 : f16 to f32
          %29 = arith.mulf %27, %28 : f32
          %30 = arith.extf %in_5 : f16 to f32
          %31 = arith.addf %29, %30 : f32
          %32 = arith.truncf %31 : f32 to f16
          linalg.yield %32 : f16
        } -> tensor<2048x1280xf16>
        flow.dispatch.tensor.store %21, %8, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
        return
      }
    }
  }
}
