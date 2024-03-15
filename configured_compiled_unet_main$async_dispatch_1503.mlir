hal.executable public @main$async_dispatch_1503 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1503_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1503_rocm_hsaco_fb_0_source.mlir> : vector<4821xi8>}) {
    hal.executable.export public @main$async_dispatch_1503_generic_2x32x20x16384_f16xf32xf32xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1503_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [1024 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1503_generic_2x32x20x16384_f16xf32xf32xf32() {
        %cst = arith.constant 9.99999974E-6 : f32
        %cst_0 = arith.constant 3.276800e+05 : f32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x20x16384xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2x32x20x16384xf32>>
        %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [2, 32, 20, 16384], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x20x16384xf16>> -> tensor<2x32x20x16384xf16>
        %7 = tensor.empty() : tensor<2x32x20x16384xf32>
        %8 = tensor.empty() : tensor<2x32xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<2x32x20x16384xf16>) outs(%7 : tensor<2x32x20x16384xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f16, %out: f32):
          %15 = arith.extf %in : f16 to f32
          linalg.yield %15 : f32
        } -> tensor<2x32x20x16384xf32>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} ins(%cst_1 : f32) outs(%8 : tensor<2x32xf32>) -> tensor<2x32xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%9 : tensor<2x32x20x16384xf32>) outs(%10 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.addf %in, %out : f32
          linalg.yield %15 : f32
        } -> tensor<2x32xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<2x32xf32>) outs(%8 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.divf %in, %cst_0 : f32
          linalg.yield %15 : f32
        } -> tensor<2x32xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%9, %12 : tensor<2x32x20x16384xf32>, tensor<2x32xf32>) outs(%10 : tensor<2x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %15 = arith.subf %in, %in_2 : f32
          %16 = arith.mulf %15, %15 : f32
          %17 = arith.addf %16, %out : f32
          linalg.yield %17 : f32
        } -> tensor<2x32xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %12, %13 : tensor<2x32x20x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%7 : tensor<2x32x20x16384xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 1, 4096]]>} {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst : f32
          %17 = math.rsqrt %16 : f32
          %18 = arith.extf %in : f16 to f32
          %19 = arith.subf %18, %in_2 : f32
          %20 = arith.mulf %19, %17 : f32
          linalg.yield %20 : f32
        } -> tensor<2x32x20x16384xf32>
        flow.dispatch.tensor.store %14, %5, offsets = [0, 0, 0, 0], sizes = [2, 32, 20, 16384], strides = [1, 1, 1, 1] : tensor<2x32x20x16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x20x16384xf32>>
        return
      }
    }
  }
}
