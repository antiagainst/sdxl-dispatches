hal.executable public @main$async_dispatch_122 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_122_conv_2d_nhwc_hwcf_2x32x32x1280x1x1x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_122_conv_2d_nhwc_hwcf_2x32x32x1280x1x1x640_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c74765824 = arith.constant 74765824 : index
        %c114846720 = arith.constant 114846720 : index
        %c117384704 = arith.constant 117384704 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c74765824) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x32x640xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c114846720) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x640x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c117384704) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x32x640xf16>> -> tensor<2x32x32x640xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 640, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x640x1280xf16>> -> tensor<1x1x640x1280xf16>
        %5 = tensor.empty() : tensor<2x32x32x1280xf32>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%5 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x32x32x640xf16>, tensor<1x1x640x1280xf16>) outs(%6 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        return
      }
    }
  }
}