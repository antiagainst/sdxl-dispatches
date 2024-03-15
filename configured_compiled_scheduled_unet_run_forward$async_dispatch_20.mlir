hal.executable public @run_forward$async_dispatch_20 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_20_generic_2x320x16384_f16xf16xf32xf32xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_20_generic_2x320x16384_f16xf16xf32xf32xf16() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>>
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x320xf32>>
        %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
        %12 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        %13 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0], sizes = [2, 16384, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16384x320xf16>> -> tensor<2x16384x320xf16>
        %14 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %15 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [2, 320], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x320xf32>> -> tensor<2x320xf32>
        %16 = flow.dispatch.tensor.load %11, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
        %17 = tensor.empty() : tensor<2x320x16384xf16>
        %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %14, %15, %16 : tensor<2x16384x320xf16>, tensor<320xf16>, tensor<2x320xf32>, tensor<320xf32>) outs(%17 : tensor<2x320x16384xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f32, %in_2: f32, %out: f16):
          %19 = arith.addf %in_1, %in_2 : f32
          %20 = arith.truncf %19 : f32 to f16
          %21 = arith.addf %in, %in_0 : f16
          %22 = arith.addf %21, %20 : f16
          linalg.yield %22 : f16
        } -> tensor<2x320x16384xf16>
        flow.dispatch.tensor.store %18, %12, offsets = [0, 0, 0], sizes = [2, 320, 16384], strides = [1, 1, 1] : tensor<2x320x16384xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x16384xf16>>
        return
      }
    }
  }
}
