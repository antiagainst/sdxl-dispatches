hal.executable public @main$async_dispatch_1524 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1524_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1524_rocm_hsaco_fb_0_source.mlir> : vector<3702xi8>}) {
    hal.executable.export public @main$async_dispatch_1524_generic_2x320x16384_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1524_rocm_hsaco_fb_0_source.mlir":9:6)}, translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1524_generic_2x320x16384_f16() {
        %c126121280 = arith.constant 126121280 : index
        %c105149760 = arith.constant 105149760 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c126121280) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c105149760) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 16384, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>> -> tensor<2x16384x320xf16>
        %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 16384, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>> -> tensor<2x16384x320xf16>
        %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %9 = tensor.empty() : tensor<2x320x16384xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7, %8 : tensor<2x16384x320xf16>, tensor<320xf16>, tensor<2x16384x320xf16>, tensor<320xf16>) outs(%9 : tensor<2x320x16384xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
          %11 = arith.addf %in_1, %in_2 : f16
          %12 = arith.addf %in, %in_0 : f16
          %13 = arith.addf %12, %11 : f16
          linalg.yield %13 : f16
        } -> tensor<2x320x16384xf16>
        flow.dispatch.tensor.store %10, %4, offsets = [0, 0, 0], sizes = [2, 320, 16384], strides = [1, 1, 1] : tensor<2x320x16384xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        return
      }
    }
  }
}
