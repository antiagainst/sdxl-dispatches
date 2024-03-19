hal.executable public @main$async_dispatch_1186 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1186_generic_2x640x4096_f32xf32xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1186_generic_2x640x4096_f32xf32xf16xf32() {
        %c95737344 = arith.constant 95737344 : index
        %c85251584 = arith.constant 85251584 : index
        %c1734364160 = arith.constant 1734364160 : index
        %c64280064 = arith.constant 64280064 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c95737344) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1734364160) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c85251584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c64280064) : !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf32>> -> tensor<2x4096x640xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>> -> tensor<2x4096x640xf16>
        %7 = tensor.empty() : tensor<2x640x4096xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5, %6 : tensor<2x4096x640xf32>, tensor<640xf32>, tensor<2x4096x640xf16>) outs(%7 : tensor<2x640x4096xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f32):
          %9 = arith.addf %in, %in_0 : f32
          %10 = arith.truncf %9 : f32 to f16
          %11 = arith.addf %10, %in_1 : f16
          %12 = arith.extf %11 : f16 to f32
          linalg.yield %12 : f32
        } -> tensor<2x640x4096xf32>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [2, 640, 4096], strides = [1, 1, 1] : tensor<2x640x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf32>>
        return
      }
    }
  }
}
