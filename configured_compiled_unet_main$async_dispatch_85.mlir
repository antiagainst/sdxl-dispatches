hal.executable public @main$async_dispatch_85 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_85_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_85_rocm_hsaco_fb_0_source.mlir> : vector<2171xi8>}) {
    hal.executable.export public @main$async_dispatch_85_transpose_2x640x4096_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_85_rocm_hsaco_fb_0_source.mlir":9:6)}, translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_85_transpose_2x640x4096_f16() {
        %c68449600 = arith.constant 68449600 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%1) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c68449600) : !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf16>>
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xf16>> -> tensor<2x4096x640xf16>
        %5 = tensor.empty() : tensor<2x640x4096xf16>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<2x4096x640xf16>) outs(%5 : tensor<2x640x4096xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<2x640x4096xf16>
        flow.dispatch.tensor.store %6, %3, offsets = [0, 0, 0], sizes = [2, 640, 4096], strides = [1, 1, 1] : tensor<2x640x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x640x4096xf16>>
        return
      }
    }
  }
}
