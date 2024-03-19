hal.executable public @main$async_dispatch_20 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_20_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_20_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c85251584 = arith.constant 85251584 : index
        %c1365504 = arith.constant 1365504 : index
        %c1960960 = arith.constant 1960960 : index
        %c1959680 = arith.constant 1959680 : index
        %c22337024 = arith.constant 22337024 : index
        %c43308544 = arith.constant 43308544 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c85251584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1960960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1365504) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1959680) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c22337024) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c43308544) : !flow.dispatch.tensor<writeonly:tensor<2x320x128x128xf16>>
        %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>> -> tensor<2x130x130x320xf16>
        %7 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 320, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>> -> tensor<3x3x320x320xf16>
        %8 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
        %9 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
        %10 = tensor.empty() : tensor<2x320x128x128xf16>
        %11 = tensor.empty() : tensor<2x128x128x320xf16>
        %12 = tensor.empty() : tensor<2x128x128x320xf32>
        %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} ins(%cst : f32) outs(%12 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %14 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>, strides = dense<1> : vector<2xi64>} ins(%6, %7 : tensor<2x130x130x320xf16>, tensor<3x3x320x320xf16>) outs(%13 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %15:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %14, %9 : tensor<2x128x128x320xf16>, tensor<2x128x128x320xf32>, tensor<320xf32>) outs(%11, %10 : tensor<2x128x128x320xf16>, tensor<2x320x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} {
        ^bb0(%in: f16, %in_0: f32, %in_1: f32, %out: f16, %out_2: f16):
          %16 = arith.addf %in_0, %in_1 : f32
          %17 = arith.truncf %16 : f32 to f16
          %18 = arith.addf %in, %17 : f16
          linalg.yield %18, %18 : f16, f16
        } -> (tensor<2x128x128x320xf16>, tensor<2x320x128x128xf16>)
        flow.dispatch.tensor.store %15#0, %4, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        flow.dispatch.tensor.store %15#1, %5, offsets = [0, 0, 0, 0], sizes = [2, 320, 128, 128], strides = [1, 1, 1, 1] : tensor<2x320x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x128x128xf16>>
        return
      }
    }
  }
}
