hal.executable public @main$async_dispatch_1055 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1055_conv_2d_nhwc_hwcf_2x64x64x1280x3x3x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1055_conv_2d_nhwc_hwcf_2x64x64x1280x3x3x1280_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c121951744 = arith.constant 121951744 : index
        %c1588014080 = arith.constant 1588014080 : index
        %c1588008960 = arith.constant 1588008960 : index
        %c144254464 = arith.constant 144254464 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c121951744) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x66x66x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1588014080) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1588008960) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c144254464) : !flow.dispatch.tensor<readwrite:tensor<1920x2x64x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 66, 66, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x66x66x1280xf16>> -> tensor<2x66x66x1280xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x1280xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %7 = tensor.empty() : tensor<1280x2x64x64xf16>
        %8 = tensor.empty() : tensor<2x64x64x1280xf32>
        %9 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} ins(%cst : f32) outs(%8 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
        %10 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>, strides = dense<1> : vector<2xi64>} ins(%4, %5 : tensor<2x66x66x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%9 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %6 : tensor<2x64x64x1280xf32>, tensor<1280xf32>) outs(%7 : tensor<1280x2x64x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} {
        ^bb0(%in: f32, %in_0: f32, %out: f16):
          %12 = arith.addf %in, %in_0 : f32
          %13 = arith.truncf %12 : f32 to f16
          linalg.yield %13 : f16
        } -> tensor<1280x2x64x64xf16>
        flow.dispatch.tensor.store %11, %3, offsets = [0, 0, 0, 0], sizes = [1280, 2, 64, 64], strides = [1, 1, 1, 1] : tensor<1280x2x64x64xf16> -> !flow.dispatch.tensor<readwrite:tensor<1920x2x64x64xf16>>
        return
      }
    }
  }
}
