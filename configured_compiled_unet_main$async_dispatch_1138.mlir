hal.executable public @main$async_dispatch_1138 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1138_generic_1280x2x64x64_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [64 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1138_generic_1280x2x64x64_f16() {
        %c89421120 = arith.constant 89421120 : index
        %c0 = arith.constant 0 : index
        %c110392640 = arith.constant 110392640 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c89421120) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64x64x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c110392640) : !flow.dispatch.tensor<readwrite:tensor<1920x2x64x64xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x64x1280xf16>> -> tensor<2x64x64x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf16>> -> tensor<1280xf16>
        %5 = tensor.empty() : tensor<1280x2x64x64xf16>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<2x64x64x1280xf16>, tensor<1280xf16>) outs(%5 : tensor<1280x2x64x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 2, 0]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %7 = arith.addf %in, %in_0 : f16
          linalg.yield %7 : f16
        } -> tensor<1280x2x64x64xf16>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [1280, 2, 64, 64], strides = [1, 1, 1, 1] : tensor<1280x2x64x64xf16> -> !flow.dispatch.tensor<readwrite:tensor<1920x2x64x64xf16>>
        return
      }
    }
  }
}
