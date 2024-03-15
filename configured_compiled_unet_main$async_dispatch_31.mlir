hal.executable public @main$async_dispatch_31 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_31_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_31_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c126121280 = arith.constant 126121280 : index
        %c42235200 = arith.constant 42235200 : index
        %c5575680 = arith.constant 5575680 : index
        %c0 = arith.constant 0 : index
        %c11520 = arith.constant 11520 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c126121280) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c5575680) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c42235200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c11520) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>> -> tensor<2x130x130x320xf16>
        %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 320, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>> -> tensor<3x3x320x320xf16>
        %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
        %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %9 = tensor.empty() : tensor<2x128x128x320xf16>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} ins(%cst : f16) outs(%9 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        %11 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<2x130x130x320xf16>, tensor<3x3x320x320xf16>) outs(%10 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %11, %8 : tensor<2x128x128x320xf16>, tensor<2x128x128x320xf16>, tensor<320xf16>) outs(%9 : tensor<2x128x128x320xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
          %13 = arith.addf %in_0, %in_1 : f16
          %14 = arith.addf %in, %13 : f16
          linalg.yield %14 : f16
        } -> tensor<2x128x128x320xf16>
        flow.dispatch.tensor.store %12, %4, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        return
      }
    }
  }
}
