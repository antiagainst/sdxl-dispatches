hal.executable public @main$async_dispatch_635 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_635_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_635_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c94664000 = arith.constant 94664000 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x2560xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x2560x1280xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c94664000) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf16>>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 2560], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x2560xf16>> -> tensor<2x34x34x2560xf16>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [3, 3, 2560, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x2560x1280xf16>> -> tensor<3x3x2560x1280xf16>
        %9 = tensor.empty() : tensor<2x32x32x1280xf16>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f16) outs(%9 : tensor<2x32x32x1280xf16>) -> tensor<2x32x32x1280xf16>
        %11 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%7, %8 : tensor<2x34x34x2560xf16>, tensor<3x3x2560x1280xf16>) outs(%10 : tensor<2x32x32x1280xf16>) -> tensor<2x32x32x1280xf16>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf16>>
        return
      }
    }
  }
}
