hal.executable public @run_forward$async_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @run_forward$async_dispatch_0_elementwise ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], translation_info = #iree_codegen.translation_info<LLVMGPUBaseLowering>, workgroup_size = [1 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @run_forward$async_dispatch_0_elementwise() {
        %c0_i64 = arith.constant 0 : i64
        %c31_i64 = arith.constant 31 : i64
        %cst = arith.constant dense_resource<torch_tensor_31_torch.int64> : tensor<31xi64>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
        %3 = tensor.empty() : tensor<i64>
        %4 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2 : tensor<i64>) outs(%3 : tensor<i64>) {
        ^bb0(%in: i64, %out: i64):
          %5 = arith.cmpi slt, %in, %c0_i64 : i64
          %6 = arith.addi %in, %c31_i64 : i64
          %7 = arith.select %5, %6, %in : i64
          %8 = arith.index_cast %7 : i64 to index
          %extracted = tensor.extract %cst[%8] : tensor<31xi64>
          linalg.yield %extracted : i64
        } -> tensor<i64>
        flow.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
        return
      }
    }
  }
}
