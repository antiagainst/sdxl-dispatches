hal.executable public @_initializer_240_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @_initializer_240_dispatch_0_transpose_9x2560x1280_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_initializer_240_dispatch_0_transpose_9x2560x1280_f16() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x2560x9xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%1) : !flow.dispatch.tensor<writeonly:tensor<9x2560x1280xf16>>
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [1280, 2560, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x2560x9xf16>> -> tensor<1280x2560x9xf16>
        %5 = tensor.empty() : tensor<9x2560x1280xf16>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<1280x2560x9xf16>) outs(%5 : tensor<9x2560x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 256]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<9x2560x1280xf16>
        flow.dispatch.tensor.store %6, %3, offsets = [0, 0, 0], sizes = [9, 2560, 1280], strides = [1, 1, 1] : tensor<9x2560x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<9x2560x1280xf16>>
        return
      }
    }
  }
}
