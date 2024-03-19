hal.executable public @main$async_dispatch_38 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_38_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 7, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUConvVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4, subgroup_m_tile_count = 4, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [256 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_38_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640_f16xf16xf32() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = hal.interface.constant.load[6] : i32
        %7 = arith.index_castui %0 : i32 to index
        %8 = arith.index_castui %1 : i32 to index
        %9 = arith.index_castui %2 : i32 to index
        %10 = arith.index_castui %3 : i32 to index
        %11 = arith.index_castui %4 : i32 to index
        %12 = arith.index_castui %5 : i32 to index
        %13 = arith.index_castui %6 : i32 to index
        %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x66x66x640xf16>>
        %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x640x640xf16>>
        %16 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64x64x640xf32>>
        %17 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %18 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
        %19 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        %20 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%13) : !flow.dispatch.tensor<writeonly:tensor<2x640x64x64xf16>>
        %21 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0, 0], sizes = [2, 66, 66, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x66x66x640xf16>> -> tensor<2x66x66x640xf16>
        %22 = flow.dispatch.tensor.load %15, offsets = [0, 0, 0, 0], sizes = [3, 3, 640, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x640x640xf16>> -> tensor<3x3x640x640xf16>
        %23 = flow.dispatch.tensor.load %16, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x64x640xf32>> -> tensor<2x64x64x640xf32>
        %24 = flow.dispatch.tensor.load %17, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %25 = flow.dispatch.tensor.load %18, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
        %26 = tensor.empty() : tensor<2x640x64x64xf16>
        %27 = tensor.empty() : tensor<2x64x64x640xf16>
        %28 = tensor.empty() : tensor<2x64x64x640xf32>
        %29 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 128, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} ins(%cst : f32) outs(%28 : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
        %30 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 128, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>, strides = dense<1> : vector<2xi64>} ins(%21, %22 : tensor<2x66x66x640xf16>, tensor<3x3x640x640xf16>) outs(%29 : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
        %31:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %24, %30, %25 : tensor<2x64x64x640xf32>, tensor<640xf32>, tensor<2x64x64x640xf32>, tensor<640xf32>) outs(%27, %26 : tensor<2x64x64x640xf16>, tensor<2x640x64x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 128, 0, 0, 32], [0, 0, 0, 0, 1, 1, 0]]>} {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16, %out_3: f16):
          %32 = arith.addf %in_1, %in_2 : f32
          %33 = arith.addf %in, %in_0 : f32
          %34 = arith.truncf %32 : f32 to f16
          %35 = arith.truncf %33 : f32 to f16
          %36 = arith.addf %35, %34 : f16
          linalg.yield %36, %36 : f16, f16
        } -> (tensor<2x64x64x640xf16>, tensor<2x640x64x64xf16>)
        flow.dispatch.tensor.store %31#0, %19, offsets = [0, 0, 0, 0], sizes = [2, 64, 64, 640], strides = [1, 1, 1, 1] : tensor<2x64x64x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x64x64x640xf16>>
        flow.dispatch.tensor.store %31#1, %20, offsets = [0, 0, 0, 0], sizes = [2, 640, 64, 64], strides = [1, 1, 1, 1] : tensor<2x640x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x640x64x64xf16>>
        return
      }
    }
  }
}
