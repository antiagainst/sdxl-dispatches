hal.executable public @main$async_dispatch_1332 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1332_generic_2x128x128x960_f32xf16xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1332_generic_2x128x128x960_f32xf16xf16xf16() {
        %cst = arith.constant 1.000000e+00 : f16
        %c190109184 = arith.constant 190109184 : index
        %c0 = arith.constant 0 : index
        %c315938304 = arith.constant 315938304 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c190109184) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x960x128x128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<960xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<960xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c315938304) : !flow.dispatch.tensor<readwrite:tensor<2x130x130x960xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 960, 128, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x960x128x128xf32>> -> tensor<2x960x128x128xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [960], strides = [1] : !flow.dispatch.tensor<readonly:tensor<960xf16>> -> tensor<960xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [960], strides = [1] : !flow.dispatch.tensor<readonly:tensor<960xf16>> -> tensor<960xf16>
        %7 = tensor.empty() : tensor<2x128x128x960xf16>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %5, %6 : tensor<2x960x128x128xf32>, tensor<960xf16>, tensor<960xf16>) outs(%7 : tensor<2x128x128x960xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 512]]>} {
        ^bb0(%in: f32, %in_0: f16, %in_1: f16, %out: f16):
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %in, %9 : f32
          %11 = arith.extf %in_1 : f16 to f32
          %12 = arith.addf %10, %11 : f32
          %13 = arith.truncf %12 : f32 to f16
          %14 = arith.negf %13 : f16
          %15 = math.exp %14 : f16
          %16 = arith.addf %15, %cst : f16
          %17 = arith.divf %cst, %16 : f16
          %18 = arith.mulf %17, %13 : f16
          linalg.yield %18 : f16
        } -> tensor<2x128x128x960xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 1, 1, 0], sizes = [2, 128, 128, 960], strides = [1, 1, 1, 1] : tensor<2x128x128x960xf16> -> !flow.dispatch.tensor<readwrite:tensor<2x130x130x960xf16>>
        return
      }
    }
  }
}
