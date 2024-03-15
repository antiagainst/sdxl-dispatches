hal.executable public @main$async_dispatch_142 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_142_attention_40x1024x64xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, {"amdgpu-waves-per-eu" = 2 : i64}>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_142_attention_40x1024x64xf16() {
        %cst = arith.constant 1.250000e-01 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>>
        %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>>
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>>
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<40x1024x64xf16>>
        %12 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0], sizes = [40, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>> -> tensor<40x1024x64xf16>
        %13 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [40, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>> -> tensor<40x1024x64xf16>
        %14 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [40, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<40x1024x64xf16>> -> tensor<40x1024x64xf16>
        %15 = tensor.empty() : tensor<40x1024x64xf16>
        %16 = iree_linalg_ext.attention ins(%12, %13, %14, %cst : tensor<40x1024x64xf16>, tensor<40x1024x64xf16>, tensor<40x1024x64xf16>, f16) outs(%15 : tensor<40x1024x64xf16>) -> tensor<40x1024x64xf16>
        flow.dispatch.tensor.store %16, %11, offsets = [0, 0, 0], sizes = [40, 1024, 64], strides = [1, 1, 1] : tensor<40x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<40x1024x64xf16>>
        return
      }
    }
  }
}
