hal.executable public @main$async_dispatch_1433 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1433_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1433_rocm_hsaco_fb_0_source.mlir> : vector<2196xi8>}) {
    hal.executable.export public @main$async_dispatch_1433_transpose_320x2x64x64_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1433_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [64 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1433_transpose_320x2x64x64_f16() {
        %c63206720 = arith.constant 63206720 : index
        %c68449600 = arith.constant 68449600 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c63206720) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x320x64x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c68449600) : !flow.dispatch.tensor<readwrite:tensor<960x2x64x64xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 320, 64, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x320x64x64xf16>> -> tensor<2x320x64x64xf16>
        %3 = tensor.empty() : tensor<320x2x64x64xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x320x64x64xf16>) outs(%3 : tensor<320x2x64x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 2, 0]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<320x2x64x64xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [640, 0, 0, 0], sizes = [320, 2, 64, 64], strides = [1, 1, 1, 1] : tensor<320x2x64x64xf16> -> !flow.dispatch.tensor<readwrite:tensor<960x2x64x64xf16>>
        return
      }
    }
  }
}
