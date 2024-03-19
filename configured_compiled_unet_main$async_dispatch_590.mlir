hal.executable public @main$async_dispatch_590 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_590_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_590_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c40960 = arith.constant 40960 : index
        %c69522944 = arith.constant 69522944 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x2560xf16>>
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x2560x1280xf16>>
        %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %11 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c40960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>>
        %12 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %13 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c69522944) : !flow.dispatch.tensor<writeonly:tensor<2x1280x32x32xf16>>
        %14 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 2560], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x2560xf16>> -> tensor<2x34x34x2560xf16>
        %15 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [3, 3, 2560, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x2560x1280xf16>> -> tensor<3x3x2560x1280xf16>
        %16 = flow.dispatch.tensor.load %10, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %17 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280xf32>> -> tensor<2x1280xf32>
        %18 = flow.dispatch.tensor.load %12, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %19 = tensor.empty() : tensor<2x1280x32x32xf16>
        %20 = tensor.empty() : tensor<2x32x32x1280xf32>
        %21 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} ins(%cst : f32) outs(%20 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>, strides = dense<1> : vector<2xi64>} ins(%14, %15 : tensor<2x34x34x2560xf16>, tensor<3x3x2560x1280xf16>) outs(%21 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %16, %17, %18 : tensor<2x32x32x1280xf32>, tensor<1280xf32>, tensor<2x1280xf32>, tensor<1280xf32>) outs(%19 : tensor<2x1280x32x32xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
          %24 = arith.addf %in_1, %in_2 : f32
          %25 = arith.addf %in, %in_0 : f32
          %26 = arith.truncf %24 : f32 to f16
          %27 = arith.truncf %25 : f32 to f16
          %28 = arith.addf %27, %26 : f16
          linalg.yield %28 : f16
        } -> tensor<2x1280x32x32xf16>
        flow.dispatch.tensor.store %23, %13, offsets = [0, 0, 0, 0], sizes = [2, 1280, 32, 32], strides = [1, 1, 1, 1] : tensor<2x1280x32x32xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1280x32x32xf16>>
        return
      }
    }
  }
}
