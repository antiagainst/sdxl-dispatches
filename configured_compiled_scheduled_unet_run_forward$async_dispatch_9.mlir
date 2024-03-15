hal.executable public @run_forward$async_dispatch_9 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_9_elementwise_5632_f32xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_9_elementwise_5632_f32xf16() {
        %c268608 = arith.constant 268608 : index
        %c291136 = arith.constant 291136 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c268608) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5632xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c291136) : !flow.dispatch.tensor<writeonly:tensor<5632xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [5632], strides = [1] : !flow.dispatch.tensor<readonly:tensor<5632xf32>> -> tensor<5632xf32>
        %3 = tensor.empty() : tensor<5632xf16>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<5632xf32>) outs(%3 : tensor<5632xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128]]>} {
        ^bb0(%in: f32, %out: f16):
          %5 = arith.truncf %in : f32 to f16
          linalg.yield %5 : f16
        } -> tensor<5632xf16>
        flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [5632], strides = [1] : tensor<5632xf16> -> !flow.dispatch.tensor<writeonly:tensor<5632xf16>>
        return
      }
    }
  }
}
