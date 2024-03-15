hal.executable public @main$async_dispatch_13 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_13_generic_2x320x16384_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_13_generic_2x320x16384_f16() {
        %c292160 = arith.constant 292160 : index
        %c0 = arith.constant 0 : index
        %c21263680 = arith.constant 21263680 : index
        %c42235200 = arith.constant 42235200 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c292160) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c21263680) : !flow.dispatch.tensor<writeonly:tensor<2x16384x320xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c42235200) : !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 16384, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>> -> tensor<2x16384x320xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %6 = tensor.empty() : tensor<2x16384x320xf16>
        %7 = tensor.empty() : tensor<2x320x16384xf16>
        %8:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5 : tensor<2x16384x320xf16>, tensor<320xf16>) outs(%6, %7 : tensor<2x16384x320xf16>, tensor<2x320x16384xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: f16):
          %9 = arith.addf %in, %in_0 : f16
          linalg.yield %9, %9 : f16, f16
        } -> (tensor<2x16384x320xf16>, tensor<2x320x16384xf16>)
        flow.dispatch.tensor.store %8#0, %2, offsets = [0, 0, 0], sizes = [2, 16384, 320], strides = [1, 1, 1] : tensor<2x16384x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x16384x320xf16>>
        flow.dispatch.tensor.store %8#1, %3, offsets = [0, 0, 0], sizes = [2, 320, 16384], strides = [1, 1, 1] : tensor<2x320x16384xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        return
      }
    }
  }
}