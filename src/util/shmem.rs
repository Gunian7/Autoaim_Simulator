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
/// 8       8      timestamp_ns (i64, CLOCK_MONOTONIC)
/// 16      N      pixel data (RGB8, row-major, width*height*3 bytes)
///
/// Total size = 16 + width * height * 3
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

    /// Write a frame. `data` must be `width * height * 3` bytes of RGB8.
    pub fn write_frame(&self, data: &[u8], timestamp_ns: i64) {
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

        // Write timestamp
        unsafe {
            ptr::write((self.ptr as *mut i64).add(1), timestamp_ns);
        }

        // Write pixel data
        unsafe {
            let dst = self.ptr.add(16);
            ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        // Post semaphore to signal consumer
        unsafe {
            libc::sem_post(sem);
        }
    }

    fn frame_size(width: u32, height: u32) -> usize {
        16 + (width as usize) * (height as usize) * 3
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
