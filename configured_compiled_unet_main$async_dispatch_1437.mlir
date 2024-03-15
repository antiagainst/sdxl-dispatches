hal.executable public @main$async_dispatch_1437 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1437_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1437_rocm_hsaco_fb_0_source.mlir> : vector<2736xi8>}) {
    hal.executable.export public @main$async_dispatch_1437_conv_2d_nhwc_hwcf_2x64x64x640x3x3x960_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1437_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 4, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1437_conv_2d_nhwc_hwcf_2x64x64x640x3x3x960_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c131364160 = arith.constant 131364160 : index
        %c599091200 = arith.constant 599091200 : index
        %c84178240 = arith.constant 84178240 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c131364160) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x66x66x960xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c599091200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x960x640xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c84178240) : !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 66, 66, 960], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x66x66x960xf16>> -> tensor<2x66x66x960xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 960, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x960x640xf16>> -> tensor<3x3x960x640xf16>
        %5 = tensor.empty() : tensor<2x64x64x640xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>} ins(%cst : f16) outs(%5 : tensor<2x64x64x640xf16>) -> tensor<2x64x64x640xf16>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x66x66x960xf16>, tensor<3x3x960x640xf16>) outs(%6 : tensor<2x64x64x640xf16>) -> tensor<2x64x64x640xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 640], strides = [1, 1, 1, 1] : tensor<2x64x64x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        return
      }
    }
  }
}
