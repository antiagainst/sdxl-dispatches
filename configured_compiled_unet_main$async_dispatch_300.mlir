hal.executable public @main$async_dispatch_300 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_300_broadcast_2048x1280_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_300_broadcast_2048x1280_f16() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = arith.index_castui %0 : i32 to index
        %4 = arith.index_castui %1 : i32 to index
        %5 = arith.index_castui %2 : i32 to index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf16>>
        %9 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
        %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %11 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %12 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf16>> -> tensor<1280xf16>
        %13 = tensor.empty() : tensor<2048x1280xf16>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %11, %12 : tensor<2048x1280xf16>, tensor<2048x1280xf16>, tensor<1280xf16>) outs(%13 : tensor<2048x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 256]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
          %15 = arith.addf %in_0, %in_1 : f16
          %16 = arith.addf %in, %15 : f16
          linalg.yield %16 : f16
        } -> tensor<2048x1280xf16>
        flow.dispatch.tensor.store %14, %9, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
        return
      }
    }
  }
}
