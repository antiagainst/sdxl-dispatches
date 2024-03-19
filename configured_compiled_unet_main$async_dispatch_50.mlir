hal.executable public @main$async_dispatch_50 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_50_attention_20x4096x64xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, {"amdgpu-waves-per-eu" = 2 : i64}>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_50_attention_20x4096x64xf16() {
        %cst = arith.constant 1.250000e-01 : f16
        %c437504 = arith.constant 437504 : index
        %c273664 = arith.constant 273664 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c437504) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x64x64xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c273664) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x64x64xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %9 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [20, 64, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x64x64xf16>> -> tensor<20x64x64xf16>
        %10 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [20, 64, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x64x64xf16>> -> tensor<20x64x64xf16>
        %11 = tensor.empty() : tensor<20x4096x64xf16>
        %12 = iree_linalg_ext.attention ins(%8, %9, %10, %cst : tensor<20x4096x64xf16>, tensor<20x64x64xf16>, tensor<20x64x64xf16>, f16) outs(%11 : tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
        flow.dispatch.tensor.store %12, %7, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        return
      }
    }
  }
}
