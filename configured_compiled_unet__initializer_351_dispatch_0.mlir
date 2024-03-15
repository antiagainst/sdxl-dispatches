hal.executable public @_initializer_351_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet__initializer_351_dispatch_0_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet__initializer_351_dispatch_0_rocm_hsaco_fb_0_source.mlir> : vector<2087xi8>}) {
    hal.executable.export public @_initializer_351_dispatch_0_transpose_9x320x4_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], source_locs = {"0_source" = loc("compiled_unet__initializer_351_dispatch_0_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_initializer_351_dispatch_0_transpose_9x320x4_f16() {
        %c0 = arith.constant 0 : index
        %c643393280 = arith.constant 643393280 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x320x9xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c643393280) : !flow.dispatch.tensor<writeonly:tensor<9x320x4xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 320, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x320x9xf16>> -> tensor<4x320x9xf16>
        %3 = tensor.empty() : tensor<9x320x4xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<4x320x9xf16>) outs(%3 : tensor<9x320x4xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<9x320x4xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [9, 320, 4], strides = [1, 1, 1] : tensor<9x320x4xf16> -> !flow.dispatch.tensor<writeonly:tensor<9x320x4xf16>>
        return
      }
    }
  }
}
