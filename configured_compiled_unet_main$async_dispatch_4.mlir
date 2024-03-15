hal.executable public @main$async_dispatch_4 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_4_generic_128x12_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_4_generic_128x12_f32() {
        %c6400 = arith.constant 6400 : index
        %c18688 = arith.constant 18688 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c6400) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12x128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c18688) : !flow.dispatch.tensor<writeonly:tensor<128x12xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [12, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128xf32>> -> tensor<12x128xf32>
        %3 = tensor.empty() : tensor<128x12xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<12x128xf32>) outs(%3 : tensor<128x12xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128]]>} {
        ^bb0(%in: f32, %out: f32):
          %5 = math.cos %in : f32
          linalg.yield %5 : f32
        } -> tensor<128x12xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [128, 12], strides = [1, 1] : tensor<128x12xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x12xf32>>
        return
      }
    }
  }
}
