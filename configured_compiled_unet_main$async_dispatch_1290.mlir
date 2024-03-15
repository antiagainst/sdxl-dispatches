hal.executable public @main$async_dispatch_1290 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1290_generic_320x2x128x128_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1290_generic_320x2x128x128_f16() {
        %c11520 = arith.constant 11520 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c11520) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readwrite:tensor<640x2x128x128xf16>>
        %9 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
        %10 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %11 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
        %12 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %13 = tensor.empty() : tensor<320x2x128x128xf16>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %10, %11, %12 : tensor<2x128x128x320xf16>, tensor<320xf16>, tensor<2x128x128x320xf16>, tensor<320xf16>) outs(%13 : tensor<320x2x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
          %15 = arith.addf %in_1, %in_2 : f16
          %16 = arith.addf %in, %in_0 : f16
          %17 = arith.addf %16, %15 : f16
          linalg.yield %17 : f16
        } -> tensor<320x2x128x128xf16>
        flow.dispatch.tensor.store %14, %8, offsets = [0, 0, 0, 0], sizes = [320, 2, 128, 128], strides = [1, 1, 1, 1] : tensor<320x2x128x128xf16> -> !flow.dispatch.tensor<readwrite:tensor<640x2x128x128xf16>>
        return
      }
    }
  }
}
