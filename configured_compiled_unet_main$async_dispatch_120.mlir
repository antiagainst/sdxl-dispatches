hal.executable public @main$async_dispatch_120 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_120_generic_2x32x40x1024_f16xf32xf32xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [1024 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_120_generic_2x32x40x1024_f16xf32xf32xf32() {
        %c69522944 = arith.constant 69522944 : index
        %cst = arith.constant 9.99999974E-6 : f32
        %cst_0 = arith.constant 4.096000e+04 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c69522944) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x40x1024xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%1) : !flow.dispatch.tensor<writeonly:tensor<2x32x40x1024xf32>>
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 40, 1024], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x40x1024xf16>> -> tensor<2x32x40x1024xf16>
        %5 = tensor.empty() : tensor<2x32x40x1024xf32>
        %6 = tensor.empty() : tensor<2x32xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<2x32x40x1024xf16>) outs(%5 : tensor<2x32x40x1024xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} {
        ^bb0(%in: f16, %out: f32):
          %13 = arith.extf %in : f16 to f32
          linalg.yield %13 : f32
        } -> tensor<2x32x40x1024xf32>
        %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} ins(%cst_1 : f32) outs(%6 : tensor<2x32xf32>) -> tensor<2x32xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7 : tensor<2x32x40x1024xf32>) outs(%8 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.addf %in, %out : f32
          linalg.yield %13 : f32
        } -> tensor<2x32xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2x32xf32>) outs(%6 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.divf %in, %cst_0 : f32
          linalg.yield %13 : f32
        } -> tensor<2x32xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7, %10 : tensor<2x32x40x1024xf32>, tensor<2x32xf32>) outs(%8 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %13 = arith.subf %in, %in_2 : f32
          %14 = arith.mulf %13, %13 : f32
          %15 = arith.addf %14, %out : f32
          linalg.yield %15 : f32
        } -> tensor<2x32xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %10, %11 : tensor<2x32x40x1024xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%5 : tensor<2x32x40x1024xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4, 1024]]>} {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
          %13 = arith.divf %in_3, %cst_0 : f32
          %14 = arith.addf %13, %cst : f32
          %15 = math.rsqrt %14 : f32
          %16 = arith.extf %in : f16 to f32
          %17 = arith.subf %16, %in_2 : f32
          %18 = arith.mulf %17, %15 : f32
          linalg.yield %18 : f32
        } -> tensor<2x32x40x1024xf32>
        flow.dispatch.tensor.store %12, %3, offsets = [0, 0, 0, 0], sizes = [2, 32, 40, 1024], strides = [1, 1, 1, 1] : tensor<2x32x40x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x40x1024xf32>>
        return
      }
    }
  }
}
