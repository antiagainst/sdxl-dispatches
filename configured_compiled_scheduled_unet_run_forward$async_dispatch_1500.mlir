hal.executable public @run_forward$async_dispatch_1500 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_1500_contract_2x128x128x320x960_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_1500_contract_2x128x128x320x960_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %c105411968 = arith.constant 105411968 : index
        %c63468928 = arith.constant 63468928 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320x960xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c105411968) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<960x2x128x128xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c63468928) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [320, 960], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<320x960xf16>> -> tensor<320x960xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [960, 2, 128, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<960x2x128x128xf16>> -> tensor<960x2x128x128xf16>
        %5 = tensor.empty() : tensor<2x128x128x320xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 64]]>} ins(%cst : f16) outs(%5 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<320x960xf16>, tensor<960x2x128x128xf16>) outs(%6 : tensor<2x128x128x320xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in_0, %in : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<2x128x128x320xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        return
      }
    }
  }
}
