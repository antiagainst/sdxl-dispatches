hal.executable public @main$async_dispatch_94 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_94_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_94_rocm_hsaco_fb_0_source.mlir> : vector<3130xi8>}) {
    hal.executable.export public @main$async_dispatch_94_broadcast_8192x640_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_94_rocm_hsaco_fb_0_source.mlir":9:6)}, subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_94_broadcast_8192x640_f16() {
        %c78935360 = arith.constant 78935360 : index
        %c68449600 = arith.constant 68449600 : index
        %c0 = arith.constant 0 : index
        %c89421120 = arith.constant 89421120 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c78935360) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c68449600) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c89421120) : !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %7 = tensor.empty() : tensor<8192x640xf16>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5, %6 : tensor<8192x640xf16>, tensor<8192x640xf16>, tensor<640xf16>) outs(%7 : tensor<8192x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 512]]>} {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
          %9 = arith.addf %in_0, %in_1 : f16
          %10 = arith.addf %in, %9 : f16
          linalg.yield %10 : f16
        } -> tensor<8192x640xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : tensor<8192x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        return
      }
    }
  }
}
