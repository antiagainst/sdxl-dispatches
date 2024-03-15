hal.executable public @run_forward$async_dispatch_169 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_169_elementwise_2x1024x5120_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_169_elementwise_2x1024x5120_f16() {
        %cst = arith.constant 5.000000e-01 : f16
        %cst_0 = arith.constant 1.000000e+00 : f16
        %cst_1 = arith.constant 2.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2x1024x5120xf16>>
        %6 = tensor.empty() : tensor<2x1024x5120xf16>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 5120], sizes = [2, 1024, 5120], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>> -> tensor<2x1024x5120xf16>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [2, 1024, 5120], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>> -> tensor<2x1024x5120xf16>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %7 : tensor<2x1024x5120xf16>, tensor<2x1024x5120xf16>) outs(%6 : tensor<2x1024x5120xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512]]>} {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %10 = math.sqrt %cst_1 : f16
          %11 = arith.divf %in_2, %10 : f16
          %12 = math.erf %11 : f16
          %13 = arith.addf %12, %cst_0 : f16
          %14 = arith.mulf %13, %cst : f16
          %15 = arith.mulf %in_2, %14 : f16
          %16 = arith.mulf %in, %15 : f16
          linalg.yield %16 : f16
        } -> tensor<2x1024x5120xf16>
        flow.dispatch.tensor.store %9, %5, offsets = [0, 0, 0], sizes = [2, 1024, 5120], strides = [1, 1, 1] : tensor<2x1024x5120xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1024x5120xf16>>
        return
      }
    }
  }
}
