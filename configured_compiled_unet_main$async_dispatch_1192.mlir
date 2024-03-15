hal.executable public @main$async_dispatch_1192 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1192_contract_2x64x64x640x1280_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 2, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1192_contract_2x64x64x640x1280_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %c131364160 = arith.constant 131364160 : index
        %c78935360 = arith.constant 78935360 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c131364160) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x2x64x64xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c78935360) : !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [640, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<640x1280xf16>> -> tensor<640x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1280, 2, 64, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x2x64x64xf16>> -> tensor<1280x2x64x64xf16>
        %5 = tensor.empty() : tensor<2x64x64x640xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 64]]>} ins(%cst : f16) outs(%5 : tensor<2x64x64x640xf16>) -> tensor<2x64x64x640xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<640x1280xf16>, tensor<1280x2x64x64xf16>) outs(%6 : tensor<2x64x64x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in_0, %in : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<2x64x64x640xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 640], strides = [1, 1, 1, 1] : tensor<2x64x64x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        return
      }
    }
  }
}
