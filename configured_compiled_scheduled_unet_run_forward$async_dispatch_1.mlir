hal.executable public @run_forward$async_dispatch_1 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_1_generic_160x2_i64xf32xf32xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_1_generic_160x2_i64xf32xf32xf32() {
        %c0 = arith.constant 0 : index
        %c17536 = arith.constant 17536 : index
        %c262208 = arith.constant 262208 : index
        %c264768 = arith.constant 264768 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c17536) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<160xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c262208) : !flow.dispatch.tensor<readwrite:tensor<320x2xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c264768) : !flow.dispatch.tensor<writeonly:tensor<160x2xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
        %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [160], strides = [1] : !flow.dispatch.tensor<readonly:tensor<160xf32>> -> tensor<160xf32>
        %6 = tensor.empty() : tensor<160x2xf32>
        %7:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : tensor<i64>, tensor<160xf32>) outs(%6, %6 : tensor<160x2xf32>, tensor<160x2xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128]]>} {
        ^bb0(%in: i64, %in_0: f32, %out: f32, %out_1: f32):
          %8 = arith.sitofp %in : i64 to f32
          %9 = arith.mulf %8, %in_0 : f32
          %10 = math.sin %9 : f32
          %11 = math.cos %9 : f32
          linalg.yield %10, %11 : f32, f32
        } -> (tensor<160x2xf32>, tensor<160x2xf32>)
        flow.dispatch.tensor.store %7#0, %2, offsets = [0, 0], sizes = [160, 2], strides = [1, 1] : tensor<160x2xf32> -> !flow.dispatch.tensor<readwrite:tensor<320x2xf32>>
        flow.dispatch.tensor.store %7#1, %3, offsets = [0, 0], sizes = [160, 2], strides = [1, 1] : tensor<160x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<160x2xf32>>
        return
      }
    }
  }
}
