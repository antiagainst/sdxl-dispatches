hal.executable public @run_forward$async_dispatch_1529 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_1529_generic_4x128x128_i64xf16xf16xf16xf16xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_1529_generic_4x128x128_i64xf16xf16xf16xf16xf16() {
        %c33_i64 = arith.constant 33 : i64
        %c967_i64 = arith.constant 967 : i64
        %c0_i64 = arith.constant 0 : i64
        %cst = arith.constant 0.999149978 : f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %cst_1 = arith.constant 5.000000e-01 : f32
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %c18176 = arith.constant 18176 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c18176) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1000xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x128x128xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<f16>>
        %5 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x128x128xf16>>
        %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [1000], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1000xf32>> -> tensor<1000xf32>
        %7 = flow.dispatch.tensor.load %2, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
        %8 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x128x128xf16>> -> tensor<4x128x128xf16>
        %9 = flow.dispatch.tensor.load %4, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f16>> -> tensor<f16>
        %10 = tensor.empty() : tensor<4x128x128xf16>
        %11 = flow.dispatch.tensor.load %0, offsets = [1, 0, 0, 0], sizes = [1, 128, 128, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>> -> tensor<128x128x4xf16>
        %12 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 128, 128, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x128x4xf16>> -> tensor<128x128x4xf16>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %8, %12, %9, %11 : tensor<i64>, tensor<4x128x128xf16>, tensor<128x128x4xf16>, tensor<f16>, tensor<128x128x4xf16>) outs(%10 : tensor<4x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128]]>} {
        ^bb0(%in: i64, %in_2: f16, %in_3: f16, %in_4: f16, %in_5: f16, %out: f16):
          %14 = arith.subi %in, %c33_i64 : i64
          %15 = arith.addi %in, %c967_i64 : i64
          %16 = arith.cmpi sge, %14, %c0_i64 : i64
          %17 = arith.select %16, %14, %15 : i64
          %18 = arith.index_cast %17 : i64 to index
          %extracted = tensor.extract %6[%18] : tensor<1000xf32>
          %19 = arith.index_cast %in : i64 to index
          %extracted_6 = tensor.extract %6[%19] : tensor<1000xf32>
          %20 = arith.select %16, %extracted, %cst : f32
          %21 = arith.divf %20, %extracted_6 : f32
          %22 = arith.subf %cst_0, %20 : f32
          %23 = math.powf %22, %cst_1 : f32
          %24 = arith.subf %cst_0, %extracted_6 : f32
          %25 = arith.mulf %extracted_6, %24 : f32
          %26 = arith.mulf %25, %20 : f32
          %27 = math.powf %26, %cst_1 : f32
          %28 = arith.mulf %extracted_6, %23 : f32
          %29 = math.powf %21, %cst_1 : f32
          %30 = arith.extf %in_2 : f16 to f32
          %31 = arith.mulf %29, %30 : f32
          %32 = arith.subf %in_5, %in_3 : f16
          %33 = arith.mulf %in_4, %32 : f16
          %34 = arith.addf %in_3, %33 : f16
          %35 = arith.subf %20, %extracted_6 : f32
          %36 = arith.extf %34 : f16 to f32
          %37 = arith.mulf %35, %36 : f32
          %38 = arith.addf %28, %27 : f32
          %39 = arith.divf %37, %38 : f32
          %40 = arith.subf %31, %39 : f32
          %41 = arith.truncf %40 : f32 to f16
          linalg.yield %41 : f16
        } -> tensor<4x128x128xf16>
        flow.dispatch.tensor.store %13, %5, offsets = [0, 0, 0], sizes = [4, 128, 128], strides = [1, 1, 1] : tensor<4x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<4x128x128xf16>>
        return
      }
    }
  }
}
