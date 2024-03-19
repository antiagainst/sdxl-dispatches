hal.executable public @main$async_dispatch_1013 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1013_transpose_2048x1920_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1013_transpose_2048x1920_f16() {
        %c116708864 = arith.constant 116708864 : index
        %c69522944 = arith.constant 69522944 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c116708864) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1920x2048xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c69522944) : !flow.dispatch.tensor<writeonly:tensor<2048x1920xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1920, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1920x2048xf16>> -> tensor<1920x2048xf16>
        %3 = tensor.empty() : tensor<2048x1920xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1920x2048xf16>) outs(%3 : tensor<2048x1920xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<2048x1920xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [2048, 1920], strides = [1, 1] : tensor<2048x1920xf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x1920xf16>>
        return
      }
    }
  }
}
