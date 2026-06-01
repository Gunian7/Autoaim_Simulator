use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

// Safety: ShmFrameWriter only uses raw pointers and fds inside the same process.
// It is Send+Sync because all operations are synchronized via a named semaphore.
unsafe impl Send for ShmFrameWriter {}
unsafe impl Sync for ShmFrameWriter {}

/// Layout of the shared memory frame:
///
/// offset  size   field
/// 0       4      width  (u32)
/// 4       4      height (u32)
/// 8       8      timestamp_ns (i64)
/// 16      4      gimbal_yaw   (f32, rad)
/// 20      4      gimbal_pitch (f32, rad)
/// 24      4      gimbal_roll  (f32, rad)
/// 28      4      bullet_speed (f32)
/// 32      4      mode         (i32: 0=idle,1=auto_aim,2=small_buff,3=big_buff,4=outpost)
/// 36      N      pixel data (RGB8, row-major, width*height*3 bytes)
///
/// Total size = 36 + width * height * 3
pub struct ShmFrameWriter {
    fd: i32,
    ptr: *mut u8,
    map_size: usize,
    width: u32,
    height: u32,
    closed: AtomicBool,
}

static SHM_NAME: &str = "/simulator_frame";
static SEM_NAME: &str = "/simulator_sem";

impl ShmFrameWriter {
    /// Create (or open+resize) the shared memory region for a given resolution.
    pub fn open(width: u32, height: u32) -> Self {
        let map_size = Self::frame_size(width, height);

        // Create the semaphore first (do not destroy it on drop – let the
        // consumer close/unlink it when it wants).
        let sem_name = CString::new(SEM_NAME).unwrap();
        let sem = unsafe { libc::sem_open(sem_name.as_ptr(), libc::O_CREAT, 0o644, 0) };
        assert!(
            !sem.is_null() && sem != libc::SEM_FAILED,
            "sem_open failed"
        );
        // We keep the semaphore open for the entire lifetime. The consumer
        // side should open the same semaphore separately.
        // Set initial value to 1 so the writer can write immediately.
        unsafe {
            libc::sem_post(sem);
        }

        let shm_name = CString::new(SHM_NAME).unwrap();
        let fd = unsafe {
            let fd = libc::shm_open(
                shm_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                0o644,
            );
            assert!(fd >= 0, "shm_open failed for {}", SHM_NAME);
            // Set size
            let ret = libc::ftruncate(fd, map_size as i64);
            assert_eq!(ret, 0, "ftruncate failed");
            fd
        };

        let ptr = unsafe {
            let p = libc::mmap(
                ptr::null_mut(),
                map_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            assert!(p != libc::MAP_FAILED, "mmap failed");
            p as *mut u8
        };

        // Write header (width, height)
        unsafe {
            ptr::write(ptr as *mut u32, width);
            ptr::write((ptr as *mut u32).add(1), height);
        }

        Self {
            fd,
            ptr,
            map_size,
            width,
            height,
            closed: AtomicBool::new(false),
        }
    }

    /// Gimbal metadata layout offset constants.
    const GIMBAL_YAW_OFFSET: usize   = 16; // f32
    const GIMBAL_PITCH_OFFSET: usize = 20; // f32
    const GIMBAL_ROLL_OFFSET: usize  = 24; // f32
    const BULLET_SPEED_OFFSET: usize = 28; // f32
    const MODE_OFFSET: usize         = 32; // i32
    const PIXEL_OFFSET: usize        = 36; // pixel data starts

    /// Write a frame. `data` must be `width * height * 3` bytes of RGB8.
    /// `gimbal_yaw/pitch/roll` in radians, `bullet_speed` in m/s,
    /// `mode` matches io::Mode enum (0=idle,1=auto_aim,...).
    pub fn write_frame(
        &self,
        data: &[u8],
        timestamp_ns: i64,
        gimbal_yaw: f32,
        gimbal_pitch: f32,
        gimbal_roll: f32,
        bullet_speed: f32,
        mode: i32,
    ) {
        if self.closed.load(Ordering::Relaxed) {
            return;
        }
        let expected_len = self.width as usize * self.height as usize * 3;
        assert_eq!(
            data.len(),
            expected_len,
            "ShmFrameWriter: data size mismatch"
        );

        // Wait for semaphore (consumer signals when it has finished reading)
        let sem_name = CString::new(SEM_NAME).unwrap();
        let sem = unsafe { libc::sem_open(sem_name.as_ptr(), 0) };
        assert!(!sem.is_null() && sem != libc::SEM_FAILED);
        unsafe {
            libc::sem_wait(sem);
        }

        unsafe {
            // Timestamp (offset 8)
            ptr::write((self.ptr as *mut i64).add(1), timestamp_ns);

            // Gimbal yaw/pitch/roll (offsets 16,20,24)
            let base_f32 = self.ptr.add(Self::GIMBAL_YAW_OFFSET) as *mut f32;
            ptr::write(base_f32, gimbal_yaw);
            ptr::write(base_f32.add(1), gimbal_pitch);
            ptr::write(base_f32.add(2), gimbal_roll);

            // Bullet speed (offset 28)
            ptr::write(self.ptr.add(Self::BULLET_SPEED_OFFSET) as *mut f32, bullet_speed);

            // Mode (offset 32)
            ptr::write(self.ptr.add(Self::MODE_OFFSET) as *mut i32, mode);

            // Pixel data (offset 36)
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(Self::PIXEL_OFFSET), data.len());
        }

        // Post semaphore to signal consumer
        unsafe {
            libc::sem_post(sem);
        }
    }

    fn frame_size(width: u32, height: u32) -> usize {
        36 + (width as usize) * (height as usize) * 3
    }
}

impl Drop for ShmFrameWriter {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Relaxed);
        if !self.ptr.is_null() {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.map_size);
            }
        }
        if self.fd >= 0 {
            unsafe {
                libc::close(self.fd);
            }
        }
        // Do NOT unlink shm/sem here – the consumer may still need them.
        // The C++ side will unlink on its own teardown, or we rely on
        // system cleanup on reboot. If we want to clean up, we should
        // coordinate with the consumer.
    }
}
