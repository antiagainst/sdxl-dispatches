hal.executable public @main$async_dispatch_902 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_902_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_902_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c124573184 = arith.constant 124573184 : index
        %c40960 = arith.constant 40960 : index
        %c1338388480 = arith.constant 1338388480 : index
        %c1338383360 = arith.constant 1338383360 : index
        %c1382625280 = arith.constant 1382625280 : index
        %c69522944 = arith.constant 69522944 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c124573184) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x1920xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1338388480) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x1920x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1338383360) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c40960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>>
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1382625280) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c69522944) : !flow.dispatch.tensor<writeonly:tensor<2x1280x32x32xf16>>
        %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 1920], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x1920xf16>> -> tensor<2x34x34x1920xf16>
        %7 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1920, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1920x1280xf16>> -> tensor<3x3x1920x1280xf16>
        %8 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>> -> tensor<2x1280xf32>
        %10 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %11 = tensor.empty() : tensor<2x1280x32x32xf16>
        %12 = tensor.empty() : tensor<2x32x32x1280xf32>
        %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} ins(%cst : f32) outs(%12 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %14 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>, strides = dense<1> : vector<2xi64>} ins(%6, %7 : tensor<2x34x34x1920xf16>, tensor<3x3x1920x1280xf16>) outs(%13 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %8, %9, %10 : tensor<2x32x32x1280xf32>, tensor<1280xf32>, tensor<2x1280xf32>, tensor<1280xf32>) outs(%11 : tensor<2x1280x32x32xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
          %16 = arith.addf %in_1, %in_2 : f32
          %17 = arith.addf %in, %in_0 : f32
          %18 = arith.truncf %16 : f32 to f16
          %19 = arith.truncf %17 : f32 to f16
          %20 = arith.addf %19, %18 : f16
          linalg.yield %20 : f16
        } -> tensor<2x1280x32x32xf16>
        flow.dispatch.tensor.store %15, %5, offsets = [0, 0, 0, 0], sizes = [2, 1280, 32, 32], strides = [1, 1, 1, 1] : tensor<2x1280x32x32xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1280x32x32xf16>>
        return
      }
    }
  }
}
