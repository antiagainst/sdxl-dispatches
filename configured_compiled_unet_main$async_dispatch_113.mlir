hal.executable public @main$async_dispatch_113 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
    hal.executable.export public @main$async_dispatch_113_slow_memcpy ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUDistribute>, workgroup_size = [128 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main$async_dispatch_113_slow_memcpy() {
        %c80008704 = arith.constant 80008704 : index
        %c111465984 = arith.constant 111465984 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c80008704) flags(ReadOnly) : memref<2x64x64x640xf16, strided<[2621440, 40960, 640, 1], offset: 40004352>, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %0, 64 : memref<2x64x64x640xf16, strided<[2621440, 40960, 640, 1], offset: 40004352>, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c111465984) : memref<2x66x66x640xf16, strided<[2787840, 42240, 640, 1], offset: 55732992>, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %1, 64 : memref<2x66x66x640xf16, strided<[2787840, 42240, 640, 1], offset: 55732992>, #hal.descriptor_type<storage_buffer>>
        %subview = memref.subview %1[0, 1, 1, 0] [2, 64, 64, 640] [1, 1, 1, 1] : memref<2x66x66x640xf16, strided<[2787840, 42240, 640, 1], offset: 55732992>, #hal.descriptor_type<storage_buffer>> to memref<2x64x64x640xf16, strided<[2787840, 42240, 640, 1], offset: 55775872>, #hal.descriptor_type<storage_buffer>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : memref<2x64x64x640xf16, strided<[2621440, 40960, 640, 1], offset: 40004352>, #hal.descriptor_type<storage_buffer>>) outs(%subview : memref<2x64x64x640xf16, strided<[2787840, 42240, 640, 1], offset: 55775872>, #hal.descriptor_type<storage_buffer>>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128]]>} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        }
        return
      }
    }
  }
}
