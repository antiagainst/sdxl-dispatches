hal.executable public @main$async_dispatch_9 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_9_contract_2x1280x2816_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUWarpReduction>, workgroup_size = [704 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_9_contract_2x1280x2816_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+00 : f16
        %c291072 = arith.constant 291072 : index
        %c0 = arith.constant 0 : index
        %c10240 = arith.constant 10240 : index
        %c268544 = arith.constant 268544 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c291072) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2816x2xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x2816xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c10240) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c268544) : !flow.dispatch.tensor<writeonly:tensor<2x1280xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2816, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2816x2xf16>> -> tensor<2816x2xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 2816], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x2816xf16>> -> tensor<1280x2816xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %7 = tensor.empty() : tensor<2x1280xf16>
        %8 = tensor.empty() : tensor<2x1280xf32>
        %9 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 2816]]>} ins(%cst : f32) outs(%8 : tensor<2x1280xf32>) -> tensor<2x1280xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %5 : tensor<2816x2xf16>, tensor<1280x2816xf16>) outs(%9 : tensor<2x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 2816]]>} {
        ^bb0(%in: f16, %in_1: f16, %out: f32):
          %12 = arith.extf %in : f16 to f32
          %13 = arith.extf %in_1 : f16 to f32
          %14 = arith.mulf %12, %13 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<2x1280xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %6 : tensor<2x1280xf32>, tensor<1280xf32>) outs(%7 : tensor<2x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 2816]]>} {
        ^bb0(%in: f32, %in_1: f32, %out: f16):
          %12 = arith.addf %in, %in_1 : f32
          %13 = arith.truncf %12 : f32 to f16
          %14 = arith.negf %13 : f16
          %15 = math.exp %14 : f16
          %16 = arith.addf %15, %cst_0 : f16
          %17 = arith.divf %cst_0, %16 : f16
          %18 = arith.mulf %17, %13 : f16
          linalg.yield %18 : f16
        } -> tensor<2x1280xf16>
        flow.dispatch.tensor.store %11, %3, offsets = [0, 0], sizes = [2, 1280], strides = [1, 1] : tensor<2x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1280xf16>>
        return
      }
    }
  }
}
