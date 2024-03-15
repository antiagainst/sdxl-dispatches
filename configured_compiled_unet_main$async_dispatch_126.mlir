hal.executable public @main$async_dispatch_126 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_126_generic_2x640x1024_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_126_generic_2x640x1024_f16() {
        %c89421120 = arith.constant 89421120 : index
        %c0 = arith.constant 0 : index
        %c92042560 = arith.constant 92042560 : index
        %c94664000 = arith.constant 94664000 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c89421120) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1024x640xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c92042560) : !flow.dispatch.tensor<writeonly:tensor<2x1024x640xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c94664000) : !flow.dispatch.tensor<writeonly:tensor<2x640x1024xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 1024, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x640xf16>> -> tensor<2x1024x640xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %6 = tensor.empty() : tensor<2x1024x640xf16>
        %7 = tensor.empty() : tensor<2x640x1024xf16>
        %8:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5 : tensor<2x1024x640xf16>, tensor<640xf16>) outs(%6, %7 : tensor<2x1024x640xf16>, tensor<2x640x1024xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: f16):
          %9 = arith.addf %in, %in_0 : f16
          linalg.yield %9, %9 : f16, f16
        } -> (tensor<2x1024x640xf16>, tensor<2x640x1024xf16>)
        flow.dispatch.tensor.store %8#0, %2, offsets = [0, 0, 0], sizes = [2, 1024, 640], strides = [1, 1, 1] : tensor<2x1024x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1024x640xf16>>
        flow.dispatch.tensor.store %8#1, %3, offsets = [0, 0, 0], sizes = [2, 640, 1024], strides = [1, 1, 1] : tensor<2x640x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x640x1024xf16>>
        return
      }
    }
  }
}
