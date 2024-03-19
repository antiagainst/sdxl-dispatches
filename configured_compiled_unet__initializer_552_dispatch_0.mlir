hal.executable public @_initializer_552_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @_initializer_552_dispatch_0_elementwise_4_f16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_initializer_552_dispatch_0_elementwise_4_f16xf32() {
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1280) : !flow.dispatch.tensor<readwrite:tensor<4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf16>> -> tensor<4xf16>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xf32>> -> tensor<4xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<4xf16>) outs(%3 : tensor<4xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128]]>} {
        ^bb0(%in: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          linalg.yield %5 : f32
        } -> tensor<4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4xf32>>
        return
      }
    }
  }
}
