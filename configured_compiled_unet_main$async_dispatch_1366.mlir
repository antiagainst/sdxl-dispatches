hal.executable public @main$async_dispatch_1366 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1366_generic_4x128x128_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1366_generic_4x128x128_f16() {
        %c2097152 = arith.constant 2097152 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c2097152) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<f16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x128x128xf16>>
        %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f16>> -> tensor<f16>
        %4 = tensor.empty() : tensor<4x128x128xf16>
        %5 = flow.dispatch.tensor.load %0, offsets = [1, 0, 0, 0], sizes = [1, 128, 128, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>> -> tensor<128x128x4xf16>
        %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 128, 128, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>> -> tensor<128x128x4xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %3, %5 : tensor<128x128x4xf16>, tensor<f16>, tensor<128x128x4xf16>) outs(%4 : tensor<4x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
          %8 = arith.subf %in_1, %in : f16
          %9 = arith.mulf %in_0, %8 : f16
          %10 = arith.addf %in, %9 : f16
          linalg.yield %10 : f16
        } -> tensor<4x128x128xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : tensor<4x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<4x128x128xf16>>
        return
      }
    }
  }
}
