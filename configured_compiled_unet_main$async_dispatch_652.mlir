hal.executable public @main$async_dispatch_652 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_652_generic_1280x2048_f16xf32xf32xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_652_generic_1280x2048_f16xf32xf32xf16() {
        %c111465984 = arith.constant 111465984 : index
        %c116708864 = arith.constant 116708864 : index
        %c776867840 = arith.constant 776867840 : index
        %c127194624 = arith.constant 127194624 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c111465984) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c116708864) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c776867840) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c127194624) : !flow.dispatch.tensor<writeonly:tensor<1280x2048xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf32>> -> tensor<2048x1280xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %7 = tensor.empty() : tensor<1280x2048xf16>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5, %6 : tensor<2048x1280xf16>, tensor<2048x1280xf32>, tensor<1280xf32>) outs(%7 : tensor<1280x2048xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f16, %in_0: f32, %in_1: f32, %out: f16):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.truncf %9 : f32 to f16
          %11 = arith.addf %in, %10 : f16
          linalg.yield %11 : f16
        } -> tensor<1280x2048xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [1280, 2048], strides = [1, 1] : tensor<1280x2048xf16> -> !flow.dispatch.tensor<writeonly:tensor<1280x2048xf16>>
        return
      }
    }
  }
}
