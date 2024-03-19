hal.executable public @main$async_dispatch_1339 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_1339_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_1339_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = arith.index_castui %0 : i32 to index
        %7 = arith.index_castui %1 : i32 to index
        %8 = arith.index_castui %2 : i32 to index
        %9 = arith.index_castui %3 : i32 to index
        %10 = arith.index_castui %4 : i32 to index
        %11 = arith.index_castui %5 : i32 to index
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>>
        %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>>
        %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf32>>
        %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
        %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
        %17 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%11) : !flow.dispatch.tensor<readwrite:tensor<640x2x128x128xf16>>
        %18 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x320xf16>> -> tensor<2x130x130x320xf16>
        %19 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0, 0], sizes = [3, 3, 320, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x320x320xf16>> -> tensor<3x3x320x320xf16>
        %20 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x320xf32>> -> tensor<2x128x128x320xf32>
        %21 = flow.dispatch.tensor.load %15, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
        %22 = flow.dispatch.tensor.load %16, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
        %23 = tensor.empty() : tensor<320x2x128x128xf16>
        %24 = tensor.empty() : tensor<2x128x128x320xf32>
        %25 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} ins(%cst : f32) outs(%24 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %26 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%18, %19 : tensor<2x130x130x320xf16>, tensor<3x3x320x320xf16>) outs(%25 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %21, %26, %22 : tensor<2x128x128x320xf32>, tensor<320xf32>, tensor<2x128x128x320xf32>, tensor<320xf32>) outs(%23 : tensor<320x2x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
          %28 = arith.addf %in_1, %in_2 : f32
          %29 = arith.addf %in, %in_0 : f32
          %30 = arith.truncf %28 : f32 to f16
          %31 = arith.truncf %29 : f32 to f16
          %32 = arith.addf %31, %30 : f16
          linalg.yield %32 : f16
        } -> tensor<320x2x128x128xf16>
        flow.dispatch.tensor.store %27, %17, offsets = [0, 0, 0, 0], sizes = [320, 2, 128, 128], strides = [1, 1, 1, 1] : tensor<320x2x128x128xf16> -> !flow.dispatch.tensor<readwrite:tensor<640x2x128x128xf16>>
        return
      }
    }
  }
}
