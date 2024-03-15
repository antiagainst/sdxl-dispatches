hal.executable public @main$async_dispatch_1379 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) sources({compiled_unet_main$async_dispatch_1379_rocm_hsaco_fb_0_source.mlir = dense_resource<compiled_unet_main$async_dispatch_1379_rocm_hsaco_fb_0_source.mlir> : vector<3395xi8>}) {
    hal.executable.export public @main$async_dispatch_1379_generic_640x8192_f32xf32xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], source_locs = {"0_source" = loc("compiled_unet_main$async_dispatch_1379_rocm_hsaco_fb_0_source.mlir":9:6)}, translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1379_generic_640x8192_f32xf32xf16xf16() {
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf32>>
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<640x8192xf16>>
        %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf32>> -> tensor<8192x640xf32>
        %13 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %14 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %15 = tensor.empty() : tensor<640x8192xf16>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %13, %14 : tensor<8192x640xf32>, tensor<640xf32>, tensor<8192x640xf16>) outs(%15 : tensor<640x8192xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f16):
          %17 = arith.addf %in, %in_0 : f32
          %18 = arith.truncf %17 : f32 to f16
          %19 = arith.addf %18, %in_1 : f16
          linalg.yield %19 : f16
        } -> tensor<640x8192xf16>
        flow.dispatch.tensor.store %16, %11, offsets = [0, 0], sizes = [640, 8192], strides = [1, 1] : tensor<640x8192xf16> -> !flow.dispatch.tensor<writeonly:tensor<640x8192xf16>>
        return
      }
    }
  }
}
