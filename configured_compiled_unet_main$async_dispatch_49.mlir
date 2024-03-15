hal.executable public @main$async_dispatch_49 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_49_contract_3x2x10x4096x64x640_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 2, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_49_contract_3x2x10x4096x64x640_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = arith.index_castui %0 : i32 to index
        %4 = arith.index_castui %1 : i32 to index
        %5 = arith.index_castui %2 : i32 to index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x10x64x640xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<3x2x10x4096x64xf16>>
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>> -> tensor<2x4096x640xf16>
        %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [3, 10, 64, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x10x64x640xf16>> -> tensor<3x10x64x640xf16>
        %11 = tensor.empty() : tensor<3x2x10x4096x64xf16>
        %12 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 64]]>} ins(%cst : f16) outs(%11 : tensor<3x2x10x4096x64xf16>) -> tensor<3x2x10x4096x64xf16>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : tensor<2x4096x640xf16>, tensor<3x10x64x640xf16>) outs(%12 : tensor<3x2x10x4096x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %14 = arith.mulf %in, %in_0 : f16
          %15 = arith.addf %out, %14 : f16
          linalg.yield %15 : f16
        } -> tensor<3x2x10x4096x64xf16>
        flow.dispatch.tensor.store %13, %8, offsets = [0, 0, 0, 0, 0], sizes = [3, 2, 10, 4096, 64], strides = [1, 1, 1, 1, 1] : tensor<3x2x10x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<3x2x10x4096x64xf16>>
        return
      }
    }
  }
}
