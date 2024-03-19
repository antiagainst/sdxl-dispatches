hal.executable public @main$async_dispatch_115 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_115_generic_2x1024x640_f32xf32xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_115_generic_2x1024x640_f32xf32xf16xf16() {
        %c69522944 = arith.constant 69522944 : index
        %c63214080 = arith.constant 63214080 : index
        %c74765824 = arith.constant 74765824 : index
        %c77387264 = arith.constant 77387264 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c69522944) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1024x640xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c63214080) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c74765824) : !flow.dispatch.tensor<writeonly:tensor<2x1024x640xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c77387264) : !flow.dispatch.tensor<writeonly:tensor<2x640x1024xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 1024, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x640xf32>> -> tensor<2x1024x640xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %6 = tensor.empty() : tensor<2x1024x640xf16>
        %7 = tensor.empty() : tensor<2x640x1024xf16>
        %8:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5 : tensor<2x1024x640xf32>, tensor<640xf32>) outs(%6, %7 : tensor<2x1024x640xf16>, tensor<2x640x1024xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128]]>} {
        ^bb0(%in: f32, %in_0: f32, %out: f16, %out_1: f16):
          %9 = arith.addf %in, %in_0 : f32
          %10 = arith.truncf %9 : f32 to f16
          linalg.yield %10, %10 : f16, f16
        } -> (tensor<2x1024x640xf16>, tensor<2x640x1024xf16>)
        flow.dispatch.tensor.store %8#0, %2, offsets = [0, 0, 0], sizes = [2, 1024, 640], strides = [1, 1, 1] : tensor<2x1024x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1024x640xf16>>
        flow.dispatch.tensor.store %8#1, %3, offsets = [0, 0, 0], sizes = [2, 640, 1024], strides = [1, 1, 1] : tensor<2x640x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x640x1024xf16>>
        return
      }
    }
  }
}
