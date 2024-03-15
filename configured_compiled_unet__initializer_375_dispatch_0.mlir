hal.executable public @_initializer_375_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @_initializer_375_dispatch_0_transpose_1920x1280_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_initializer_375_dispatch_0_transpose_1920x1280_f16() {
        %c0 = arith.constant 0 : index
        %c1579064320 = arith.constant 1579064320 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1920xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1579064320) : !flow.dispatch.tensor<writeonly:tensor<1920x1280xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1280, 1920], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1920xf16>> -> tensor<1280x1920xf16>
        %3 = tensor.empty() : tensor<1920x1280xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1280x1920xf16>) outs(%3 : tensor<1920x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<1920x1280xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1920, 1280], strides = [1, 1] : tensor<1920x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<1920x1280xf16>>
        return
      }
    }
  }
}
