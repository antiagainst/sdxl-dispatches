hal.executable public @main$async_dispatch_1511 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1511_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1511_rocm_hsaco_fb_0_source.mlir> : vector<3110xi8>}) {
    hal.executable.export public @main$async_dispatch_1511_contract_2x128x128x320x640_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1511_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 4>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1511_contract_2x128x128x320x640_f16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320x640xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640x2x128x128xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [320, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<320x640xf16>> -> tensor<320x640xf16>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [640, 2, 128, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<640x2x128x128xf16>> -> tensor<640x2x128x128xf16>
        %9 = tensor.empty() : tensor<2x128x128x320xf16>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 64]]>} ins(%cst : f16) outs(%9 : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%7, %8 : tensor<320x640xf16>, tensor<640x2x128x128xf16>) outs(%10 : tensor<2x128x128x320xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 64, 64]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %12 = arith.mulf %in_0, %in : f16
          %13 = arith.addf %out, %12 : f16
          linalg.yield %13 : f16
        } -> tensor<2x128x128x320xf16>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x320xf16>>
        return
      }
    }
  }
}