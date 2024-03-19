hal.executable public @main$async_dispatch_1005 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1005_transpose_640x2x32x32_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [32 : index, 4 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1005_transpose_640x2x32x32_f16() {
        %c74765824 = arith.constant 74765824 : index
        %c116708864 = arith.constant 116708864 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c74765824) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x32x32x640xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c116708864) : !flow.dispatch.tensor<readwrite:tensor<1920x2x32x32xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x32x640xf16>> -> tensor<2x32x32x640xf16>
        %3 = tensor.empty() : tensor<640x2x32x32xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x32x32x640xf16>) outs(%3 : tensor<640x2x32x32xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 4, 0]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<640x2x32x32xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [1280, 0, 0, 0], sizes = [640, 2, 32, 32], strides = [1, 1, 1, 1] : tensor<640x2x32x32xf16> -> !flow.dispatch.tensor<readwrite:tensor<1920x2x32x32xf16>>
        return
      }
    }
  }
}
