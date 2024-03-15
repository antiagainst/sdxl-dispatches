hal.executable public @main$async_dispatch_39 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_39_generic_2x640x4096_f16xf16xf32xf32xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_39_generic_2x640x4096_f16xf16xf32xf32xf16() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = arith.index_castui %0 : i32 to index
        %4 = arith.index_castui %1 : i32 to index
        %5 = arith.index_castui %2 : i32 to index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x640xf32>>
        %9 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %10 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf16>>
        %11 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>> -> tensor<2x4096x640xf16>
        %12 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %13 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [2, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x640xf32>> -> tensor<2x640xf32>
        %14 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %15 = tensor.empty() : tensor<2x640x4096xf16>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %12, %13, %14 : tensor<2x4096x640xf16>, tensor<640xf16>, tensor<2x640xf32>, tensor<640xf32>) outs(%15 : tensor<2x640x4096xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f32, %in_2: f32, %out: f16):
          %17 = arith.addf %in_1, %in_2 : f32
          %18 = arith.truncf %17 : f32 to f16
          %19 = arith.addf %in, %in_0 : f16
          %20 = arith.addf %19, %18 : f16
          linalg.yield %20 : f16
        } -> tensor<2x640x4096xf16>
        flow.dispatch.tensor.store %16, %10, offsets = [0, 0, 0], sizes = [2, 640, 4096], strides = [1, 1, 1] : tensor<2x640x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf16>>
        return
      }
    }
  }
}
