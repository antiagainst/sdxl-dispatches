hal.executable public @main$async_dispatch_41 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_41_generic_2x64x64x640_f32xf16xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_41_generic_2x64x64x640_f32xf16xf16xf16() {
        %cst = arith.constant 1.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x640x64x64xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readwrite:tensor<2x66x66x640xf16>>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [2, 640, 64, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x640x64x64xf32>> -> tensor<2x640x64x64xf32>
        %9 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %10 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %11 = tensor.empty() : tensor<2x64x64x640xf16>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %9, %10 : tensor<2x640x64x64xf32>, tensor<640xf16>, tensor<640xf16>) outs(%11 : tensor<2x64x64x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 512]]>} {
        ^bb0(%in: f32, %in_0: f16, %in_1: f16, %out: f16):
          %13 = arith.extf %in_0 : f16 to f32
          %14 = arith.mulf %in, %13 : f32
          %15 = arith.extf %in_1 : f16 to f32
          %16 = arith.addf %14, %15 : f32
          %17 = arith.truncf %16 : f32 to f16
          %18 = arith.negf %17 : f16
          %19 = math.exp %18 : f16
          %20 = arith.addf %19, %cst : f16
          %21 = arith.divf %cst, %20 : f16
          %22 = arith.mulf %21, %17 : f16
          linalg.yield %22 : f16
        } -> tensor<2x64x64x640xf16>
        flow.dispatch.tensor.store %12, %7, offsets = [0, 1, 1, 0], sizes = [2, 64, 64, 640], strides = [1, 1, 1, 1] : tensor<2x64x64x640xf16> -> !flow.dispatch.tensor<readwrite:tensor<2x66x66x640xf16>>
        return
      }
    }
  }
}
