hal.executable public @main$async_dispatch_1317 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1317_conv_2d_nhwc_hwcf_2x128x128x4x3x3x320_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1317_conv_2d_nhwc_hwcf_2x128x128x4x3x3x320_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c62914560 = arith.constant 62914560 : index
        %c1939367680 = arith.constant 1939367680 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c62914560) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1939367680) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x320x4xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x4xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>> -> tensor<2x130x130x320xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 320, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x320x4xf16>> -> tensor<3x3x320x4xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf16>> -> tensor<4xf16>
        %7 = tensor.empty() : tensor<2x128x128x4xf16>
        %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128, 4, 4, 4]]>} ins(%cst : f16) outs(%7 : tensor<2x128x128x4xf16>) -> tensor<2x128x128x4xf16>
        %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128, 4, 4, 4]]>, strides = dense<1> : vector<2xi64>} ins(%4, %5 : tensor<2x130x130x320xf16>, tensor<3x3x320x4xf16>) outs(%8 : tensor<2x128x128x4xf16>) -> tensor<2x128x128x4xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<2x128x128x4xf16>, tensor<4xf16>) outs(%7 : tensor<2x128x128x4xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128, 4, 4, 4]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.addf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<2x128x128x4xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 4], strides = [1, 1, 1, 1] : tensor<2x128x128x4xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x4xf16>>
        return
      }
    }
  }
}
