hal.executable public @main$async_dispatch_12 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_12_conv_2d_nhwc_hwcf_2x128x128x320x3x3x4_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [16 : index, 2 : index, 2 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_12_conv_2d_nhwc_hwcf_2x128x128x320x3x3x4_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c21760 = arith.constant 21760 : index
        %c20480 = arith.constant 20480 : index
        %c292160 = arith.constant 292160 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c21760) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x4xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c20480) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x4x320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c292160) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x4xf16>> -> tensor<2x130x130x4xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 4, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x4x320xf16>> -> tensor<3x3x4x320xf16>
        %5 = tensor.empty() : tensor<2x128x128x320xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 64, 1, 1, 4], [0, 1, 0, 0]]>} ins(%cst : f16) outs(%5 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 64, 1, 1, 4], [0, 1, 0, 0]]>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x130x130x4xf16>, tensor<3x3x4x320xf16>) outs(%6 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        return
      }
    }
  }
}
