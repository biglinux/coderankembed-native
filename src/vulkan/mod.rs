//! Compute shaders on a Vulkan GPU, with nothing but `std`: the system's
//! `libvulkan.so.1` is opened at run time, so a machine without it still
//! runs the CPU engine.
//!
//! [`Gpu::open`] picks a GPU with half-float arithmetic and storage, and
//! [`Gpu::run`] records dispatches of [`Kernel`]s over [`Buffer`]s into one
//! submission and waits for it. Dropping a buffer or a kernel waits until the
//! GPU is idle, so none is freed while a submission uses it.

mod ffi;
mod open;

use std::mem::ManuallyDrop;
use std::ptr::{null, null_mut};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ffi::{Handle, Object};
use open::{Instance, choose, create_device, create_instance, device_name};

/// What a submission's dispatches and copies do, for its barriers.
const GPU_STAGES: u32 = ffi::PIPELINE_STAGE_COMPUTE | ffi::PIPELINE_STAGE_TRANSFER;
const GPU_READS: u32 = ffi::ACCESS_SHADER_READ | ffi::ACCESS_TRANSFER_READ;
const GPU_WRITES: u32 = ffi::ACCESS_SHADER_WRITE | ffi::ACCESS_TRANSFER_WRITE;

/// Largest host buffer [`Gpu::upload_many`] stages through.
pub(crate) const STAGING_BYTES: usize = 32 << 20;

/// Descriptor sets one submission may use, and storage buffers per set.
const SETS: u32 = 512;
const BINDINGS: u32 = 8;

fn check(what: &str, result: i32) -> Result<(), String> {
    let name = match result {
        ffi::SUCCESS => return Ok(()),
        -1 => "out of host memory",
        -2 => "out of device memory",
        -3 => "initialization failed",
        -4 => "device lost",
        -9 => "incompatible driver",
        _ => return Err(format!("{what}: Vulkan error {result}")),
    };
    Err(format!("{what}: {name}"))
}

/// The logical device and its instance, destroyed once the last object made
/// from them is.
struct Device {
    handle: Handle,
    d: ffi::device::Functions,
    memory: ffi::MemoryProperties,
    /// Set once a failed wait, for a submission or for the device while an
    /// object drops, may have left work running. From then on nothing is
    /// submitted, no host buffer is read or written, and every object made
    /// from the device is leaked with the device and its instance, without
    /// waiting again.
    busy: AtomicBool,
    /// Set from a submission until its fence wait ends. `Gpu::run` waits
    /// for each submission it makes, so otherwise the device runs nothing of
    /// ours and an object may be destroyed without waiting for it.
    in_flight: AtomicBool,
    /// Destroyed after the device.
    instance: ManuallyDrop<Instance>,
}

// SAFETY: the handles are identifiers the driver lets any thread use; the
// commands called through a shared `Device` create or destroy distinct
// objects, which Vulkan synchronizes itself, and `Drop` runs once.
unsafe impl Send for Device {}
// SAFETY: as for Send.
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        if self.busy.load(Ordering::Acquire) || !self.idle() {
            return;
        }
        // SAFETY: every object made from the device is gone, since each holds
        // an Arc of it and none was leaked, and the device is idle; the
        // instance is dropped once, after it.
        unsafe {
            (self.d.destroy_device)(self.handle, null());
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

impl Device {
    /// Wait until the device runs nothing. False when that is unknown, as
    /// when the wait runs out of memory.
    fn idle(&self) -> bool {
        // SAFETY: the device is valid.
        let result = unsafe { (self.d.device_wait_idle)(self.handle) };
        // A lost device runs nothing more.
        matches!(result, ffi::SUCCESS | ffi::ERROR_DEVICE_LOST)
    }

    /// Whether an object made from the device may be destroyed: once the
    /// device is idle, unless it is busy. Otherwise the object is to be
    /// leaked, and so is the device.
    fn release(&self) -> bool {
        if self.busy.load(Ordering::Acquire) {
            return false;
        }
        // Otherwise each object would wait for the whole device: a hundred
        // waits when the encoder drops.
        if !self.in_flight.load(Ordering::Acquire) {
            return true;
        }
        let idle = self.idle();
        if !idle {
            self.busy.store(true, Ordering::Release);
        }
        idle
    }

    /// The first memory type in `bits` with all of `flags`, trying each set
    /// of flags in turn.
    fn memory_type(&self, bits: u32, flags: &[u32]) -> Option<u32> {
        let types = &self.memory.types[..self.memory.type_count as usize];
        flags
            .iter()
            .find_map(|&want| {
                (0..types.len())
                    .find(|&i| bits & (1 << i) != 0 && types[i].property_flags & want == want)
            })
            .map(|i| i as u32)
    }
}

/// A GPU, its queue and the objects of its submission.
pub(crate) struct Gpu {
    dev: Arc<Device>,
    queue: Handle,
    commands: Object,
    cb: Handle,
    fence: Object,
    descriptors: Object,
    name: String,
    vendor: u32,
    groups: [u32; 3],
    /// Bytes a storage buffer may span in a shader.
    storage_range: u32,
    /// Set when a fence failed to reset: still signaled, it must not go with
    /// another submission, so none is made.
    unreset_fence: bool,
}

// SAFETY: `Gpu` is not `Sync`, so every reference to it, and to the queue,
// command buffers, fences and descriptor pools it owns, lives on the thread
// that holds it.
unsafe impl Send for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        if !self.dev.release() {
            return;
        }
        // SAFETY: the device is idle, so no submission uses the fence, pool
        // or command buffer; destroying a null handle does nothing.
        unsafe {
            let d = &self.dev.d;
            (d.destroy_fence)(self.dev.handle, self.fence, null());
            (d.destroy_descriptor_pool)(self.dev.handle, self.descriptors, null());
            (d.destroy_command_pool)(self.dev.handle, self.commands, null());
        }
    }
}

impl Gpu {
    /// Open the best GPU that has Vulkan 1.2 with half-float arithmetic,
    /// 16-bit storage and workgroups of 256 invocations: a discrete one
    /// first, then an integrated one. CPU implementations such as llvmpipe
    /// are left out.
    pub(crate) fn open() -> Result<Self, String> {
        let instance = create_instance()?;
        let choice = choose(&instance)?;
        let (handle, d) = create_device(&instance, &choice)?;
        let i = &instance.i;
        // SAFETY: the physical device belongs to the instance; the call
        // writes the whole struct.
        let memory = unsafe {
            let mut memory = std::mem::MaybeUninit::<ffi::MemoryProperties>::zeroed();
            (i.get_memory_properties)(choice.handle, memory.as_mut_ptr());
            memory.assume_init()
        };
        let dev = Arc::new(Device {
            handle,
            d,
            memory,
            busy: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
            instance: ManuallyDrop::new(instance),
        });
        let limits = &choice.props.limits;
        let mut gpu = Self {
            dev,
            queue: null_mut(),
            commands: 0,
            cb: null_mut(),
            fence: 0,
            descriptors: 0,
            name: device_name(&choice.props),
            vendor: choice.props.vendor_id,
            groups: limits.max_compute_work_group_count,
            storage_range: limits.max_storage_buffer_range,
            unreset_fence: false,
        };
        let (d, h) = (&gpu.dev.d, gpu.dev.handle);
        // SAFETY: the device is valid and each create-info lives across its
        // call; on failure `gpu`'s Drop frees what was made.
        unsafe {
            (d.get_queue)(h, choice.family, 0, &mut gpu.queue);
            let command_pool_info = ffi::CommandPoolCreateInfo {
                s_type: ffi::TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: null(),
                flags: ffi::POOL_RESET_COMMAND_BUFFER,
                queue_family_index: choice.family,
            };
            check(
                "creating a command pool",
                (d.create_command_pool)(h, &command_pool_info, null(), &mut gpu.commands),
            )?;
            let command_buffer_info = ffi::CommandBufferAllocateInfo {
                s_type: ffi::TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: null(),
                pool: gpu.commands,
                level: 0,
                count: 1,
            };
            check(
                "allocating command buffers",
                (d.allocate_command_buffers)(h, &command_buffer_info, &mut gpu.cb),
            )?;
            let fence_info = ffi::FenceCreateInfo {
                s_type: ffi::TYPE_FENCE_CREATE_INFO,
                p_next: null(),
                flags: 0,
            };
            check(
                "creating a fence",
                (d.create_fence)(h, &fence_info, null(), &mut gpu.fence),
            )?;
            let size = ffi::DescriptorPoolSize {
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                count: SETS * BINDINGS,
            };
            let descriptor_pool_info = ffi::DescriptorPoolCreateInfo {
                s_type: ffi::TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: null(),
                flags: 0,
                max_sets: SETS,
                size_count: 1,
                sizes: &size,
            };
            check(
                "creating a descriptor pool",
                (d.create_descriptor_pool)(h, &descriptor_pool_info, null(), &mut gpu.descriptors),
            )?;
        }
        Ok(gpu)
    }

    /// The GPU's name as its driver gives it.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Refuse dispatches past `max` workgroups an axis, as a device with
    /// lower limits does.
    #[cfg(test)]
    pub(crate) fn cap_groups(&mut self, max: u32) {
        self.groups = self.groups.map(|g| g.min(max));
    }

    /// The PCI vendor ID: 0x1002 AMD, 0x10de NVIDIA, 0x8086 Intel.
    pub(crate) fn vendor(&self) -> u32 {
        self.vendor
    }

    /// A storage buffer of `size` bytes in the GPU's own memory.
    pub(crate) fn buffer(&self, size: usize) -> Result<Buffer, String> {
        self.check_range(size)?;
        Buffer::new(&self.dev, size, &[ffi::MEMORY_DEVICE_LOCAL, 0], false)
    }

    /// Shaders may not reach past a storage buffer's range: refuse larger
    /// buffers rather than let a dispatch run out of bounds.
    fn check_range(&self, size: usize) -> Result<(), String> {
        if size > self.storage_range as usize {
            return Err(format!(
                "a buffer of {size} bytes, more than the GPU's {}",
                self.storage_range
            ));
        }
        Ok(())
    }

    /// A storage buffer of `size` bytes the CPU can also read and write,
    /// cached on the CPU side when `read` is set: the GPU's output is read
    /// back from one.
    pub(crate) fn host_buffer(&self, size: usize, read: bool) -> Result<Buffer, String> {
        self.check_range(size)?;
        let coherent = ffi::MEMORY_HOST_VISIBLE | ffi::MEMORY_HOST_COHERENT;
        let flags: &[u32] = if read {
            &[coherent | ffi::MEMORY_HOST_CACHED, coherent]
        } else {
            &[coherent]
        };
        Buffer::new(&self.dev, size, flags, true)
    }

    /// Storage buffers in the GPU's own memory of `sizes` bytes, buffer `i`
    /// holding what `fill(i, bytes)` writes, in that order. They go through
    /// host buffers of at most [`STAGING_BYTES`], cached on the CPU side,
    /// one submission each: a host buffer and a wait per buffer cost more
    /// than the copy of a layer's weights, and one host buffer for all of
    /// them would add their size to the peak memory.
    pub(crate) fn upload_many(
        &mut self,
        sizes: &[usize],
        mut fill: impl FnMut(usize, &mut [u8]),
    ) -> Result<Vec<Buffer>, String> {
        self.usable()?;
        let buffers = sizes
            .iter()
            .map(|&size| self.buffer(size))
            .collect::<Result<Vec<_>, _>>()?;
        let mut first = 0;
        while first < sizes.len() {
            // Buffers first..end, a larger one alone.
            let mut end = first + 1;
            let mut bytes = sizes[first];
            while end < sizes.len() && bytes + sizes[end] <= STAGING_BYTES {
                bytes += sizes[end];
                end += 1;
            }
            let mut staging = self.host_buffer(bytes, true)?;
            let mapped = staging.bytes_mut();
            let mut at = 0;
            for (i, &size) in sizes.iter().enumerate().take(end).skip(first) {
                fill(i, &mut mapped[at..at + size]);
                at += size;
            }
            self.run(|c| {
                let mut at = 0;
                for (buffer, &size) in buffers[first..end].iter().zip(&sizes[first..end]) {
                    c.copy(&staging, at, buffer, size)?;
                    at += size;
                }
                Ok(())
            })?;
            first = end;
        }
        Ok(buffers)
    }

    /// A compute kernel from SPIR-V, with `bindings` storage buffers in set
    /// 0 and `push` bytes of push constants.
    pub(crate) fn kernel(&self, spirv: &[u8], bindings: u32, push: u32) -> Result<Kernel, String> {
        Kernel::new(&self.dev, spirv, bindings, push)
    }

    /// Record a submission with `record`, run it and wait until it is done.
    /// Writes by the CPU before `run` are visible to it, and its writes to
    /// host buffers are visible to the CPU after. What `record` owns is
    /// dropped once the submission has finished.
    pub(crate) fn run(
        &mut self,
        mut record: impl FnMut(&mut Commands<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.usable()?;
        match self.record(&mut record).and_then(|()| self.submit()) {
            Ok(()) => self.wait(),
            Err(e) => self.reset().and(Err(e)),
        }
    }

    /// Record the command buffer with `record`, after a barrier that orders
    /// it after the CPU's writes and every earlier submission, and before one
    /// that makes its writes visible to the CPU.
    ///
    /// `record` is borrowed rather than consumed so that a buffer or kernel
    /// it owns outlives the submission that uses it.
    fn record(
        &mut self,
        record: &mut impl FnMut(&mut Commands<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let d = &self.dev.d;
        let cb = self.cb;
        let begin = ffi::CommandBufferBeginInfo {
            s_type: ffi::TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: null(),
            flags: ffi::ONE_TIME_SUBMIT,
            inheritance: null(),
        };
        // SAFETY: the command buffer is not pending: each submission is
        // waited for before the next is recorded.
        check("recording", unsafe { (d.begin_command_buffer)(cb, &begin) })?;
        let mut commands = Commands { gpu: self, sets: 0 };
        commands.barrier_from(
            ffi::PIPELINE_STAGE_HOST | GPU_STAGES,
            ffi::ACCESS_HOST_WRITE | GPU_WRITES,
        );
        let recorded = record(&mut commands);
        // Make the last writes visible to the CPU.
        let barrier = ffi::MemoryBarrier {
            s_type: ffi::TYPE_MEMORY_BARRIER,
            p_next: null(),
            src_access: GPU_WRITES,
            dst_access: ffi::ACCESS_HOST_READ,
        };
        // SAFETY: the command buffer is recording; `barrier` lives across the
        // call.
        let ended = unsafe {
            (d.cmd_pipeline_barrier)(
                cb,
                GPU_STAGES,
                ffi::PIPELINE_STAGE_HOST,
                0,
                1,
                &barrier,
                0,
                null(),
                0,
                null(),
            );
            check("recording", (d.end_command_buffer)(cb))
        };
        recorded.and(ended)
    }

    /// Queue the recorded command buffer.
    fn submit(&self) -> Result<(), String> {
        // Recording may have dropped an object whose release left the GPU busy.
        self.usable()?;
        let submit = ffi::SubmitInfo {
            s_type: ffi::TYPE_SUBMIT_INFO,
            p_next: null(),
            wait_count: 0,
            wait_semaphores: null(),
            wait_stages: null(),
            command_buffer_count: 1,
            command_buffers: &self.cb,
            signal_count: 0,
            signal_semaphores: null(),
        };
        // Set first: a failed submit may still have queued the work.
        self.dev.in_flight.store(true, Ordering::Release);
        // SAFETY: the command buffer is recorded and its fence unsignaled;
        // `submit` lives across the call.
        check("submitting", unsafe {
            (self.dev.d.queue_submit)(self.queue, 1, &submit, self.fence)
        })
    }

    /// Wait for the submission, then make its fence and descriptor sets
    /// reusable.
    fn wait(&mut self) -> Result<(), String> {
        // Waiting again could block for good, and a busy GPU's objects are
        // leaked rather than reset.
        if self.dev.busy.load(Ordering::Acquire) {
            return Err("the GPU may still be running work".into());
        }
        let (d, h) = (&self.dev.d, self.dev.handle);
        // SAFETY: the fence belongs to the device and went with the
        // submission.
        let waited = check("waiting for the GPU", unsafe {
            (d.wait_for_fences)(h, 1, &self.fence, 1, u64::MAX)
        });
        // A failed fence wait may leave the submission running, and nothing
        // below may reset its objects until the device is idle.
        if waited.is_err() && !self.dev.idle() {
            self.dev.busy.store(true, Ordering::Release);
            return waited;
        }
        // The submission has finished, or the device is idle.
        self.dev.in_flight.store(false, Ordering::Release);
        // SAFETY: the submission has finished.
        let reset = check("resetting a fence", unsafe {
            (d.reset_fences)(h, 1, &self.fence)
        });
        self.unreset_fence |= reset.is_err();
        waited.and(reset).and(self.reset())
    }

    /// An error once work may still be running on the GPU (see
    /// `Device::busy`), whose host buffers are then not to be touched, or a
    /// fence failed to reset; either way the GPU takes no more submissions.
    pub(crate) fn usable(&self) -> Result<(), String> {
        if self.dev.busy.load(Ordering::Acquire) {
            Err("the GPU may still be running work".into())
        } else if self.unreset_fence {
            Err("a fence of the GPU failed to reset".into())
        } else {
            Ok(())
        }
    }

    /// Free the descriptor sets, whose submission has finished or never
    /// ran.
    fn reset(&self) -> Result<(), String> {
        // SAFETY: no pending submission uses the descriptor sets.
        check("resetting descriptors", unsafe {
            (self.dev.d.reset_descriptor_pool)(self.dev.handle, self.descriptors, 0)
        })
    }
}

/// A submission being recorded.
pub(crate) struct Commands<'a> {
    gpu: &'a Gpu,
    sets: u32,
}

impl Commands<'_> {
    /// Objects of another `Gpu` are unknown to this one's device.
    fn check_device(&self, dev: &Arc<Device>) -> Result<(), String> {
        if Arc::ptr_eq(dev, &self.gpu.dev) {
            Ok(())
        } else {
            Err("a buffer or kernel of another GPU".into())
        }
    }

    fn barrier_from(&mut self, stages: u32, access: u32) {
        let barrier = ffi::MemoryBarrier {
            s_type: ffi::TYPE_MEMORY_BARRIER,
            p_next: null(),
            src_access: access,
            dst_access: GPU_READS | GPU_WRITES,
        };
        // SAFETY: the command buffer is recording; `barrier` lives across the call.
        unsafe {
            (self.gpu.dev.d.cmd_pipeline_barrier)(
                self.gpu.cb,
                stages,
                GPU_STAGES,
                0,
                1,
                &barrier,
                0,
                null(),
                0,
                null(),
            );
        }
    }

    /// Make the dispatches and copies recorded so far finish, and their
    /// writes visible, before the next ones start.
    pub(crate) fn barrier(&mut self) {
        self.barrier_from(GPU_STAGES, GPU_WRITES);
    }

    /// Copy `bytes` of `src` from byte `src_at` on to the start of `dst`.
    pub(crate) fn copy(
        &mut self,
        src: &Buffer,
        src_at: usize,
        dst: &Buffer,
        bytes: usize,
    ) -> Result<(), String> {
        self.check_device(&src.dev)?;
        self.check_device(&dst.dev)?;
        if src_at.checked_add(bytes).is_none_or(|end| end > src.size) || bytes > dst.size {
            return Err(format!(
                "copying {bytes} bytes from byte {src_at} of {} into {}",
                src.size, dst.size
            ));
        }
        // Vulkan forbids empty and overlapping regions; both copies change
        // nothing.
        if bytes == 0 || src.buffer == dst.buffer {
            return Ok(());
        }
        let region = ffi::BufferCopy {
            src_offset: src_at as u64,
            dst_offset: 0,
            size: bytes as u64,
        };
        // SAFETY: the command buffer is recording; both buffers are borrowed
        // until `Gpu::run` has waited for the submission.
        unsafe {
            (self.gpu.dev.d.cmd_copy_buffer)(self.gpu.cb, src.buffer, dst.buffer, 1, &region)
        };
        Ok(())
    }

    /// Run `kernel` on `groups` workgroups, its binding `i` being
    /// `buffers[i]` and its push constants `push`.
    pub(crate) fn dispatch(
        &mut self,
        kernel: &Kernel,
        buffers: &[&Buffer],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<(), String> {
        self.check_device(&kernel.dev)?;
        for buffer in buffers {
            self.check_device(&buffer.dev)?;
        }
        if buffers.len() != kernel.bindings as usize || push.len() != kernel.push as usize {
            return Err(format!(
                "a kernel of {} buffers and {} bytes of constants given {} and {}",
                kernel.bindings,
                kernel.push,
                buffers.len(),
                push.len()
            ));
        }
        if groups.iter().zip(self.gpu.groups).any(|(&g, max)| g > max) || groups.contains(&0) {
            return Err(format!(
                "{groups:?} workgroups, at most {:?}",
                self.gpu.groups
            ));
        }
        if self.sets == SETS {
            return Err(format!("more than {SETS} dispatches in one submission"));
        }
        self.sets += 1;
        let (d, h, cb) = (&self.gpu.dev.d, self.gpu.dev.handle, self.gpu.cb);
        let descriptor_set_info = ffi::DescriptorSetAllocateInfo {
            s_type: ffi::TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: null(),
            pool: self.gpu.descriptors,
            set_count: 1,
            set_layouts: &kernel.set_layout,
        };
        let mut set = 0;
        // SAFETY: the pool has room for SETS sets of BINDINGS buffers, and
        // `bindings` is at most BINDINGS (checked by `Kernel::new`).
        check("allocating descriptors", unsafe {
            (d.allocate_descriptor_sets)(h, &descriptor_set_info, &mut set)
        })?;
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|b| ffi::DescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: ffi::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| ffi::WriteDescriptorSet {
                s_type: ffi::TYPE_WRITE_DESCRIPTOR_SET,
                p_next: null(),
                set,
                binding: i as u32,
                array_element: 0,
                count: 1,
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                image_info: null(),
                buffer_info: info,
                texel_buffer_view: null(),
            })
            .collect();
        // SAFETY: the set is new and unused; the buffers are borrowed until
        // `Gpu::run` has waited for the submission that reads them.
        unsafe {
            (d.update_descriptor_sets)(h, writes.len() as u32, writes.as_ptr(), 0, null());
            (d.cmd_bind_pipeline)(cb, ffi::BIND_POINT_COMPUTE, kernel.pipeline);
            (d.cmd_bind_descriptor_sets)(
                cb,
                ffi::BIND_POINT_COMPUTE,
                kernel.layout,
                0,
                1,
                &set,
                0,
                null(),
            );
            if !push.is_empty() {
                (d.cmd_push_constants)(
                    cb,
                    kernel.layout,
                    ffi::STAGE_COMPUTE,
                    0,
                    push.len() as u32,
                    push.as_ptr().cast(),
                );
            }
            (d.cmd_dispatch)(cb, groups[0], groups[1], groups[2]);
        }
        Ok(())
    }
}

/// A storage buffer, mapped for the CPU when made by [`Gpu::host_buffer`].
pub(crate) struct Buffer {
    dev: Arc<Device>,
    buffer: Object,
    memory: Object,
    map: *mut u8,
    size: usize,
}

// SAFETY: the mapping is plain memory owned by the buffer, reached only
// through `&self` or `&mut self`.
unsafe impl Send for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.dev.release() {
            return;
        }
        // SAFETY: the device is idle, so no submission uses the buffer, and
        // freeing the memory unmaps it.
        unsafe {
            (self.dev.d.destroy_buffer)(self.dev.handle, self.buffer, null());
            (self.dev.d.free_memory)(self.dev.handle, self.memory, null());
        }
    }
}

impl Buffer {
    fn new(dev: &Arc<Device>, size: usize, flags: &[u32], map: bool) -> Result<Self, String> {
        let (d, h) = (&dev.d, dev.handle);
        let buffer_info = ffi::BufferCreateInfo {
            s_type: ffi::TYPE_BUFFER_CREATE_INFO,
            p_next: null(),
            flags: 0,
            size: size.max(4) as u64,
            usage: ffi::BUFFER_STORAGE | ffi::BUFFER_TRANSFER_SRC | ffi::BUFFER_TRANSFER_DST,
            sharing_mode: 0,
            queue_family_count: 0,
            queue_families: null(),
        };
        let mut buffer = Self {
            dev: dev.clone(),
            buffer: 0,
            memory: 0,
            map: null_mut(),
            size,
        };
        // SAFETY: each create-info lives across its call; on failure Drop
        // frees what was made, null handles being no-ops.
        unsafe {
            check(
                "creating a buffer",
                (d.create_buffer)(h, &buffer_info, null(), &mut buffer.buffer),
            )?;
            let mut needs = ffi::MemoryRequirements::default();
            (d.get_buffer_requirements)(h, buffer.buffer, &mut needs);
            let index = dev
                .memory_type(needs.memory_type_bits, flags)
                .ok_or("no memory type for a buffer")?;
            let alloc = ffi::MemoryAllocateInfo {
                s_type: ffi::TYPE_MEMORY_ALLOCATE_INFO,
                p_next: null(),
                size: needs.size,
                memory_type_index: index,
            };
            check(
                &format!("allocating {size} bytes"),
                (d.allocate_memory)(h, &alloc, null(), &mut buffer.memory),
            )?;
            check(
                "binding memory",
                (d.bind_buffer_memory)(h, buffer.buffer, buffer.memory, 0),
            )?;
            if map {
                let mut ptr = null_mut();
                check(
                    "mapping memory",
                    (d.map_memory)(h, buffer.memory, 0, ffi::WHOLE_SIZE, 0, &mut ptr),
                )?;
                buffer.map = ptr.cast();
            }
        }
        Ok(buffer)
    }

    /// The buffer's bytes, which only a host buffer has. Panics on a device
    /// buffer, and once a failed wait left the GPU busy.
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `size` bytes and lives as long as the
        // buffer; no submission uses it while the slice lives, since
        // `Gpu::run` waits for its own or marks the device busy.
        unsafe { std::slice::from_raw_parts(self.mapping(), self.size) }
    }

    /// The buffer's bytes, to fill before a run; only a host buffer has
    /// them, and as for [`Buffer::bytes`], not while a submission uses it.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `bytes`, and `&mut self` makes the slice unique.
        unsafe { std::slice::from_raw_parts_mut(self.mapping(), self.size) }
    }

    /// The CPU mapping, refused for a device buffer, which has none, and
    /// while the GPU may still use the buffer.
    fn mapping(&self) -> *mut u8 {
        assert!(!self.map.is_null(), "a device buffer has no CPU mapping");
        assert!(
            !self.dev.busy.load(Ordering::Acquire),
            "the GPU may still use the buffer"
        );
        self.map
    }
}

/// A compute pipeline and its layout.
pub(crate) struct Kernel {
    dev: Arc<Device>,
    set_layout: Object,
    layout: Object,
    pipeline: Object,
    bindings: u32,
    push: u32,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if !self.dev.release() {
            return;
        }
        // SAFETY: the device is idle, so no submission uses the kernel; null
        // handles are no-ops.
        unsafe {
            let (d, h) = (&self.dev.d, self.dev.handle);
            (d.destroy_pipeline)(h, self.pipeline, null());
            (d.destroy_pipeline_layout)(h, self.layout, null());
            (d.destroy_set_layout)(h, self.set_layout, null());
        }
    }
}

impl Kernel {
    fn new(dev: &Arc<Device>, spirv: &[u8], bindings: u32, push: u32) -> Result<Self, String> {
        if !spirv.len().is_multiple_of(4) || spirv.get(..4) != Some(&0x0723_0203u32.to_le_bytes()) {
            return Err("not SPIR-V".into());
        }
        if bindings > BINDINGS {
            return Err(format!("{bindings} buffers, at most {BINDINGS}"));
        }
        let words: Vec<u32> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_le_bytes(w))
            .collect();
        let (d, h) = (&dev.d, dev.handle);
        let mut kernel = Self {
            dev: dev.clone(),
            set_layout: 0,
            layout: 0,
            pipeline: 0,
            bindings,
            push,
        };
        let slots: Vec<_> = (0..bindings)
            .map(|i| ffi::DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ffi::STAGE_COMPUTE,
                immutable_samplers: null(),
            })
            .collect();
        let range = ffi::PushConstantRange {
            stage_flags: ffi::STAGE_COMPUTE,
            offset: 0,
            size: push,
        };
        // SAFETY: each create-info and what it points to lives across its
        // call; on failure Drop frees what was made.
        unsafe {
            let descriptor_set_layout_info = ffi::DescriptorSetLayoutCreateInfo {
                s_type: ffi::TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: null(),
                flags: 0,
                binding_count: bindings,
                bindings: slots.as_ptr(),
            };
            check(
                "creating a set layout",
                (d.create_set_layout)(
                    h,
                    &descriptor_set_layout_info,
                    null(),
                    &mut kernel.set_layout,
                ),
            )?;
            let pipeline_layout_info = ffi::PipelineLayoutCreateInfo {
                s_type: ffi::TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                p_next: null(),
                flags: 0,
                set_layout_count: 1,
                set_layouts: &kernel.set_layout,
                push_range_count: u32::from(push > 0),
                push_ranges: &range,
            };
            check(
                "creating a pipeline layout",
                (d.create_pipeline_layout)(h, &pipeline_layout_info, null(), &mut kernel.layout),
            )?;
            let shader_module_info = ffi::ShaderModuleCreateInfo {
                s_type: ffi::TYPE_SHADER_MODULE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                code_size: spirv.len(),
                code: words.as_ptr(),
            };
            let mut module = 0;
            check(
                "loading a shader",
                (d.create_shader_module)(h, &shader_module_info, null(), &mut module),
            )?;
            let compute_pipeline_info = ffi::ComputePipelineCreateInfo {
                s_type: ffi::TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                stage: ffi::PipelineShaderStageCreateInfo {
                    s_type: ffi::TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: null(),
                    flags: 0,
                    stage: ffi::STAGE_COMPUTE,
                    module,
                    name: c"main".as_ptr(),
                    specialization: null(),
                },
                layout: kernel.layout,
                base_pipeline: 0,
                base_pipeline_index: -1,
            };
            let made = (d.create_compute_pipelines)(
                h,
                0,
                1,
                &compute_pipeline_info,
                null(),
                &mut kernel.pipeline,
            );
            (d.destroy_shader_module)(h, module, null());
            check("compiling a shader", made)?;
        }
        Ok(kernel)
    }
}
