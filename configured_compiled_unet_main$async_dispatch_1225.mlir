hal.executable public @main$async_dispatch_1225 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1225_broadcast_2x128x128x4_f32xf32xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [4 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1225_broadcast_2x128x128x4_f32xf32xf16() {
        %c0 = arith.constant 0 : index
        %c1761623040 = arith.constant 1761623040 : index
        %c2097152 = arith.constant 2097152 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x16xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1761623040) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c2097152) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x4xf16>>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
        %4 = tensor.empty() : tensor<2x128x128x4xf16>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x16xf32>> -> tensor<2x128x128x4xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %3 : tensor<2x128x128x4xf32>, tensor<4xf32>) outs(%4 : tensor<2x128x128x4xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 0]]>} {
        ^bb0(%in: f32, %in_0: f32, %out: f16):
          %7 = arith.addf %in, %in_0 : f32
          %8 = arith.truncf %7 : f32 to f16
          linalg.yield %8 : f16
        } -> tensor<2x128x128x4xf16>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 4], strides = [1, 1, 1, 1] : tensor<2x128x128x4xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x4xf16>>
        return
      }
    }
  }
}
