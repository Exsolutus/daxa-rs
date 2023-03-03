use crate::{core::*, device::* ,gpu_resources::*, semaphore::*};

use anyhow::{Result, bail};
use ash::{
    vk,
    extensions::khr,
};
use raw_window_handle::{
    RawDisplayHandle,
    RawWindowHandle
};

use std::{
    borrow::Cow,
    cell::{
        Cell,
        RefCell
    },
    ops::Deref,
    slice,
};



pub type FormatSelector = fn(vk::Format) -> i32;

#[inline] 
pub fn default_format_selector(format: vk::Format) -> i32 {
    match format {
        vk::Format::R8G8B8A8_SRGB => 90,
        vk::Format::R8G8B8A8_UNORM => 80,
        vk::Format::B8G8R8A8_SRGB => 70,
        vk::Format::B8G8R8A8_UNORM => 60,
        _ => 0
    }
}



#[derive(Clone, Debug)]
pub struct SwapchainInfo {
    pub raw_display_handle: RawDisplayHandle,
    pub raw_window_handle: RawWindowHandle,
    pub surface_format_selector: FormatSelector,
    pub present_mode: vk::PresentModeKHR,
    pub present_transform: vk::SurfaceTransformFlagsKHR,
    pub image_usage: vk::ImageUsageFlags,
    pub max_frames_in_flight: usize,
    pub debug_name: Cow<'static, str>
}

impl Default for SwapchainInfo {
    fn default() -> Self {
        Self {
            raw_display_handle: RawDisplayHandle::Windows(raw_window_handle::WindowsDisplayHandle::empty()),
            raw_window_handle: RawWindowHandle::Win32(raw_window_handle::Win32WindowHandle::empty()),
            surface_format_selector: default_format_selector,
            present_mode: vk::PresentModeKHR::FIFO,
            present_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            image_usage: vk::ImageUsageFlags::empty(),
            max_frames_in_flight: 2,
            debug_name: "".into()
        }
    }
}


/// I (pahrens) am going to document the internals here as wsi is really confusing and strange in vulkan.
/// Every frame we get a swapchain image index. This index can be non sequential in the case of mail box presentation and other modes.
/// This means we need to acquire a new index every frame to know what swapchain image to use.
///
/// IMPORTANT INFORMATION REGARDING SEMAPHORES IN WSI:
/// Binary semaphore in acquire MUST be un-signaled when recording actions ON THE CPU!
///
/// We need two binary semaphores here:
/// The acquire semaphore and the present semaphore.
/// The present semaphore is signaled in the last submission that uses the swapchain image and waited on in the present.
/// The acquire semaphore is signaled when the swapchain image is ready to be used.
/// This also means that the previous presentation of the image is finished and the semaphore used in the present is un-signaled.
/// Unfortunately there is NO other way to know when a present finishes (or the corresponding semaphore is un-signaled).
/// This means that in order to be able to reuse binary semaphores used in presentation,
/// one MUST pair them with the image they are used to present.
///
/// One can then rely on the acquire semaphore of the image beeing signaled to indicate that the present semaphore is able to be reused,
/// As a swapchain images acquire sema is signaled when it is available and its previous present is completed.
///
/// In order to reuse the the acquire semaphore we must set a limit in frames in flight and wait on the cpu to limit the frames in flight.
/// When we have this wait in place we can safely reuse the acquire semaphores with a linearly increasing index corresponding to the frame.
/// This means the acquire semaphores are not tied to the number of swapchain images like present semaphores but to the number of frames in flight!!
///
/// To limit the frames in flight we employ a timeline semaphore that must be signaled in a submission that uses or after one that uses the swapchain image.
pub struct Swapchain {
    device: Device,
    info: SwapchainInfo,

    pub(crate) api: khr::Swapchain,
    pub(crate) swapchain_handle: Cell<vk::SwapchainKHR>,
    
    surface_api: khr::Surface,
    surface_handle: Cell<vk::SurfaceKHR>,
    surface_format: vk::SurfaceFormatKHR,
    surface_extent: Cell<vk::Extent2D>,
    images: RefCell<Vec<ImageId>>,

    acquire_semaphores: RefCell<Vec<BinarySemaphore>>,
    present_semaphores: RefCell<Vec<BinarySemaphore>>,
    // Monotonically increasing frame index.
    cpu_frame_timeline: Cell<usize>,
    // cpu_frame_timeline % frames in flight. Used to index the acquire semaphores.
    acquire_semaphore_index: Cell<usize>,
    // Gpu timeline semaphore used to track how far behind the gpu is.
    // Used to limit frames in flight.
    gpu_frame_timeline: TimelineSemaphore,
    // This is the swapchain image index that acquire returns. This is not necessarily linear.
    // This index must be used for present semaphores as they are paired to the images.
    pub(crate) current_image_index: Cell<u32>
}

// impl Deref for Swapchain {
//     type Target = vk::SwapchainKHR;

//     fn deref(&self) -> &Self::Target {
//         &self.swapchain_handle
//     }
// }

// Swapchain creation methods
impl Swapchain {
    pub(crate) fn new(device: Device, info: SwapchainInfo) -> Result<Self> {
        let mut swapchain = Swapchain {
            device: device.clone(),
            info: info.clone(),
            api: khr::Swapchain::new(&device.0.context, &device.0.logical_device),
            swapchain_handle: Default::default(),
            surface_api: khr::Surface::new(&device.0.context.0.entry, &device.0.context),
            surface_handle: Default::default(),
            surface_format: Default::default(),
            surface_extent: Default::default(),
            images: Default::default(),
            acquire_semaphores: Default::default(),
            present_semaphores: Default::default(),
            cpu_frame_timeline: Default::default(),
            acquire_semaphore_index: Default::default(),
            gpu_frame_timeline: TimelineSemaphore::new(device.clone(), TimelineSemaphoreInfo{
                initial_value: 0,
                debug_name: format!("{} gpu timeline", info.debug_name).into()
            }).expect("TimelineSemaphore should be created."),
            current_image_index: Default::default()
        };

        swapchain.recreate_surface();

        let surface_formats = unsafe {
            swapchain.surface_api.get_physical_device_surface_formats(device.0.physical_device, swapchain.surface_handle.get())
                .expect("Supported formats should be queried for surface.")
        };
        debug_assert!(surface_formats.len() > 0, "No formats found.");

        let format_comparator = |a: &&vk::SurfaceFormatKHR, b: &&vk::SurfaceFormatKHR| -> std::cmp::Ordering {
            (info.surface_format_selector)(a.format).cmp(&(info.surface_format_selector)(b.format))
        };

        let best_format = surface_formats.iter()
            .max_by(format_comparator)
            .unwrap();

        swapchain.surface_format = *best_format;

        swapchain.recreate();

        // We have an acquire semaphore for each frame in flight.
        for i in 0..info.max_frames_in_flight {
            swapchain.acquire_semaphores.borrow_mut().push(BinarySemaphore::new(
                device.clone(),
                BinarySemaphoreInfo {
                    debug_name: format!("{}, image {} acquire semaphore", info.debug_name, i).into(),
                }
            ).unwrap())
        }
        // We have a present semaphore for each swapchain image.
        for i in 0..swapchain.images.borrow().len() {
            swapchain.present_semaphores.borrow_mut().push(BinarySemaphore::new(
                device.clone(),
                BinarySemaphoreInfo {
                    debug_name: format!("{}, image {} present semaphore", info.debug_name, i).into(),
                }
            ).unwrap())
        }

        Ok(swapchain)
    }
}

// Swapchain usage methods
impl Swapchain {
    #[inline]
    pub fn info(&self) -> &SwapchainInfo {
        &self.info
    }

    #[inline]
    pub fn get_surface_extent(&self) -> vk::Extent2D {
        self.surface_extent.get()
    }

    #[inline]
    pub fn get_format(&self) -> vk::Format {
        self.surface_format.format
    }

    #[inline]
    pub fn resize(&self) {
        self.recreate()
    }

    /// The ImageId may change between calls. This must be called to obtain a new swapchain image to be used for rendering.
    /// 
    /// Returns a swapchain image, that will be ready to render to when the acquire semaphore is signaled. This may return an empty image id if the swapchain is out of date.
    pub fn acquire_next_image(&self) -> ImageId {
        // A new frame starts.
        // We wait until the gpu timeline is frames of flight behind our cpu timeline value.
        // This will limit the frames in flight.
        self.gpu_frame_timeline.wait_for_value(
            (self.cpu_frame_timeline.get() as i64 - self.info.max_frames_in_flight as i64).max(0) as u64,
            u64::MAX
        );
        self.acquire_semaphore_index.set((self.cpu_frame_timeline.get() + 1) % self.info.max_frames_in_flight);

        let acquire_semaphore = &self.acquire_semaphores.borrow()[self.acquire_semaphore_index.get()];

        let result = unsafe {
            self.api.acquire_next_image(
                self.swapchain_handle.get(),
                u64::MAX,
                acquire_semaphore.0.semaphore,
                vk::Fence::null()
            )
        };

        match result {
            Ok((id, suboptimal)) => {
                self.current_image_index.set(id);
                self.cpu_frame_timeline.set(self.cpu_frame_timeline.get() + 1);
                self.images.borrow()[self.current_image_index.get() as usize]
            },
            Err(error) => {
                // The swapchain needs recreation, we can only return a null ImageId here.
                ImageId::default()
            }
        }
    }
    
    /// The acquire semaphore must be waited on in the first submission that uses the last acquired image.
    /// This semaphore may change between acquires, so it needs to be re-queried after every get_acquire_semaphore call.
    /// 
    /// Returns the binary semaphore that is signaled when the last acquired image is ready to be used.
    pub fn get_acquire_semaphore(&self) -> BinarySemaphore {
        self.acquire_semaphores.borrow()[self.acquire_semaphore_index.get() as usize].clone()
    }
    
    /// The present semaphore must be signaled in the last submission that uses the last acquired swapchain image.
    /// The present semaphore must be waited on in the present of the last acquired image.
    /// This semaphore may change between acquires, so it needs to be re-queried after every get_acquire_semaphore call.
    /// 
    /// Returns the present semaphore that needs to be signaled and waited on for present of the last acquired image.
    pub fn get_present_semaphore(&self) -> BinarySemaphore {
        self.present_semaphores.borrow()[self.current_image_index.get() as usize].clone()
    }
    
    /// The swapchain needs to know when the last use of the swapchain happens to limit the frames in flight.
    /// In the last submission that uses the swapchain image, signal this timeline semaphore with the cpu timeline value.
    /// 
    /// Returns the gpu timeline semaphore that needs to be signaled.
    pub fn get_gpu_timeline_semaphore(&self) -> TimelineSemaphore {
        self.gpu_frame_timeline.clone()
    }
    
    /// The last submission that uses the swapchain image needs to signal the timeline with the cpu value.
    /// 
    /// Returns the cpu frame timeline value.
    pub fn get_cpu_timeline_value(&self) -> usize {
        self.cpu_frame_timeline.get()
    }

    // pub fn change_present_mode(&self, present_mode: vk::PresentModeKHR) {
    //     todo!()
    // }
}

// Swapchain internal methods
impl Swapchain {
    fn get_index_of_image(&self, image: ImageId) -> Result<usize> {
        for (index, id) in self.images.borrow().iter().enumerate() {
            if *id == image {
                return Ok(index);
            }
        }

        bail!("Image does not belong to the swapchain.");
    }

    fn recreate(&self) {
        let surface_capabilities = unsafe { 
            self.surface_api.get_physical_device_surface_capabilities(self.device.0.physical_device, self.surface_handle.get())
                .expect("Physical device should provide surface capabilities.")
        };

        let surface_extent = self.surface_extent.get();
        if  surface_extent.width == surface_capabilities.current_extent.width &&
            surface_extent.height == surface_capabilities.current_extent.height &&
            self.swapchain_handle.get() != vk::SwapchainKHR::null()
        {
            return
        }

        self.surface_extent.set(vk::Extent2D {
            width: surface_capabilities.current_extent.width,
            height: surface_capabilities.current_extent.height
        });

        #[cfg(target_os = "linux")]
        {
            // TODO: I (grundlett) am too lazy to find out why the other present modes
            // fail on Linux. This can be inspected by Linux people and they can
            // submit a PR if they find a fix.
            self.info.present_mode = vk::PresentModeKHR::IMMEDIATE
        }

        let old_swapchain = self.swapchain_handle.get();

        self.device.wait_idle();

        self.cleanup();

        let usage = self.info.image_usage | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let swapchain_ci = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface_handle.get())
            .min_image_count(3)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(self.surface_extent.get())
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(slice::from_ref(&self.device.0.main_queue_family))
            .pre_transform(self.info.present_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.info.present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        self.swapchain_handle.set(unsafe {
            self.api.create_swapchain(&swapchain_ci, None)
                .expect("Swapchain should be created.")
        });

        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe { self.api.destroy_swapchain(old_swapchain, None) };
        }

        self.images.replace(unsafe {
            self.api.get_swapchain_images(self.swapchain_handle.get())
                .expect("Swapchain should provide the swapchain images.")
                .iter()
                .enumerate()
                .map(|(i, image)| {
                    let image_info = ImageInfo {
                        format: self.surface_format.format,
                        size: self.surface_extent.get().into(),
                        usage,
                        debug_name: format!("{} Image #{}", self.info.debug_name, i).into(),
                        ..Default::default()
                    };
                    self.device.0.new_swapchain_image(*image, self.surface_format.format, i as u32, usage, image_info)
                        .expect("Swapchain image should be added as gpu resource.")
                })
                .collect()
        });
    }

    fn cleanup(&self) {
        for image in self.images.borrow().iter() {
            self.device.0.zombify_image(*image);
        }
        self.images.borrow_mut().clear();
    }

    fn recreate_surface(&self) {
        if self.surface_handle.get() != vk::SurfaceKHR::null() {
            unsafe { self.surface_api.destroy_surface(self.surface_handle.get(), None) };
        }

        self.surface_handle.set(unsafe {
            let context = &self.device.0.context;
            ash_window::create_surface(
                &context.0.entry,
                &context.0.instance,
                self.info.raw_display_handle,
                self.info.raw_window_handle,
                None
            ).expect("Surface should be created for window.")
        });
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.cleanup();

        unsafe {
            self.api.destroy_swapchain(self.swapchain_handle.get(), None);
            self.surface_api.destroy_surface(self.surface_handle.get(), None);
        }
    }
}