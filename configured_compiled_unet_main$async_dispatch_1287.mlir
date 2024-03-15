hal.executable public @main$async_dispatch_1287 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1287_generic_2x32x30x16384_f16xf32xf32xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [1024 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1287_generic_2x32x30x16384_f16xf32xf32xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.915200e+05 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c168064320 = arith.constant 168064320 : index
        %c230978880 = arith.constant 230978880 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c168064320) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x30x16384xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c230978880) : !flow.dispatch.tensor<writeonly:tensor<2x32x30x16384xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 30, 16384], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x30x16384xf16>> -> tensor<2x32x30x16384xf16>
        %3 = tensor.empty() : tensor<2x32x30x16384xf32>
        %4 = tensor.empty() : tensor<2x32xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x32x30x16384xf16>) outs(%3 : tensor<2x32x30x16384xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f16, %out: f32):
          %11 = arith.extf %in : f16 to f32
          linalg.yield %11 : f32
        } -> tensor<2x32x30x16384xf32>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} ins(%cst : f32) outs(%4 : tensor<2x32xf32>) -> tensor<2x32xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5 : tensor<2x32x30x16384xf32>) outs(%6 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.addf %in, %out : f32
          linalg.yield %11 : f32
        } -> tensor<2x32xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2x32xf32>) outs(%4 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.divf %in, %cst_0 : f32
          linalg.yield %11 : f32
        } -> tensor<2x32xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5, %8 : tensor<2x32x30x16384xf32>, tensor<2x32xf32>) outs(%6 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %11 = arith.subf %in, %in_2 : f32
          %12 = arith.mulf %11, %11 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        } -> tensor<2x32xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %8, %9 : tensor<2x32x30x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%3 : tensor<2x32x30x16384xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
          %11 = arith.divf %in_3, %cst_0 : f32
          %12 = arith.addf %11, %cst_1 : f32
          %13 = math.rsqrt %12 : f32
          %14 = arith.extf %in : f16 to f32
          %15 = arith.subf %14, %in_2 : f32
          %16 = arith.mulf %15, %13 : f32
          linalg.yield %16 : f32
        } -> tensor<2x32x30x16384xf32>
        flow.dispatch.tensor.store %10, %1, offsets = [0, 0, 0, 0], sizes = [2, 32, 30, 16384], strides = [1, 1, 1, 1] : tensor<2x32x30x16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x30x16384xf32>>
        return
      }
    }
  }
}
