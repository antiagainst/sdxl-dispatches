hal.executable public @main$async_dispatch_1230 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1230_generic_640x8192_f32xf32xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1230_generic_640x8192_f32xf32xf16xf16() {
        %c69522944 = arith.constant 69522944 : index
        %c100980224 = arith.constant 100980224 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c100980224) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c69522944) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<640x8192xf16>>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf32>> -> tensor<8192x640xf32>
        %9 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %11 = tensor.empty() : tensor<640x8192xf16>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9, %10 : tensor<8192x640xf32>, tensor<640xf32>, tensor<8192x640xf16>) outs(%11 : tensor<640x8192xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f16):
          %13 = arith.addf %in, %in_0 : f32
          %14 = arith.truncf %13 : f32 to f16
          %15 = arith.addf %14, %in_1 : f16
          linalg.yield %15 : f16
        } -> tensor<640x8192xf16>
        flow.dispatch.tensor.store %12, %7, offsets = [0, 0], sizes = [640, 8192], strides = [1, 1] : tensor<640x8192xf16> -> !flow.dispatch.tensor<writeonly:tensor<640x8192xf16>>
        return
      }
    }
  }
}
