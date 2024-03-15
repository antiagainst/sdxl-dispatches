hal.executable public @main$async_dispatch_1183 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1183_transpose_2x1280x4096_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1183_transpose_2x1280x4096_f16() {
        %c131364160 = arith.constant 131364160 : index
        %c68449600 = arith.constant 68449600 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c131364160) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x2x4096xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c68449600) : !flow.dispatch.tensor<writeonly:tensor<2x1280x4096xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1280, 2, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x2x4096xf16>> -> tensor<1280x2x4096xf16>
        %3 = tensor.empty() : tensor<2x1280x4096xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1280x2x4096xf16>) outs(%3 : tensor<2x1280x4096xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<2x1280x4096xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [2, 1280, 4096], strides = [1, 1, 1] : tensor<2x1280x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1280x4096xf16>>
        return
      }
    }
  }
}
