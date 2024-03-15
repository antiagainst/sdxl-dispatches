hal.executable public @run_forward$async_dispatch_1490 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_1490_transpose_320x2x128x128_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_1490_transpose_320x2x128x128_f16() {
        %c273728 = arith.constant 273728 : index
        %c105411968 = arith.constant 105411968 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c273728) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c105411968) : !flow.dispatch.tensor<readwrite:tensor<960x2x128x128xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
        %3 = tensor.empty() : tensor<320x2x128x128xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x128x128x320xf16>) outs(%3 : tensor<320x2x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<320x2x128x128xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [640, 0, 0, 0], sizes = [320, 2, 128, 128], strides = [1, 1, 1, 1] : tensor<320x2x128x128xf16> -> !flow.dispatch.tensor<readwrite:tensor<960x2x128x128xf16>>
        return
      }
    }
  }
}
