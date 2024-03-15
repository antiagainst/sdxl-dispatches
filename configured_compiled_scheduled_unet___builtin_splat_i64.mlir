hal.executable public @__builtin_splat_i64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @__builtin_splat_i64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>], translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [1 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @__builtin_splat_i64() {
        %c0 = arith.constant 0 : index
        %c31_i64 = arith.constant 31 : i64
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?xi64>>{%c1}
        %1 = tensor.empty() : tensor<1xi64>
        %2 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[]]>} ins(%c31_i64 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
        flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1], strides = [1] : tensor<1xi64> -> !flow.dispatch.tensor<writeonly:tensor<?xi64>>{%c1}
        return
      }
    }
  }
}
