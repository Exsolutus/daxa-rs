use crate::device::*;

use anyhow::{Context as _, Result};

use ash::{
    extensions::ext::DebugUtils,
    Entry,
    Instance,
    vk,
};

use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    ops::Deref,
    os::raw::{c_char, c_void},
    sync::Arc, 
};

// Reexport
pub use vk::{
    DebugUtilsMessageSeverityFlagsEXT as MessageSeverity,
    DebugUtilsMessageTypeFlagsEXT as MessageType,
    DebugUtilsMessengerCallbackDataEXT as MessageData,
    DebugUtilsMessengerEXT as DebugUtilsMessenger
};


#[cfg(debug_assertions)]
unsafe extern "system" fn debug_utils_messenger_callback(
    message_severity: MessageSeverity,
    message_type: MessageType,
    p_message_data: *const MessageData,
    _p_user_data: *mut c_void
) -> vk::Bool32 {
    let info = std::ptr::NonNull::new(_p_user_data as *mut ContextInfo).unwrap().as_ref();
    let message = CStr::from_ptr((*p_message_data).p_message);

    let validation_callback = info.validation_callback;
    validation_callback(message_severity, message_type, message);

    vk::FALSE
}

#[cfg(debug_assertions)]
type ValidationCallback = fn(MessageSeverity, MessageType, &CStr);

#[cfg(debug_assertions)]
#[inline]
fn default_validation_callback(
    message_severity: MessageSeverity,
    message_type: MessageType,
    message: &CStr,
) {
    let mut exit = false;

    let severity = match message_severity {
        MessageSeverity::VERBOSE => "[Verbose]",
        MessageSeverity::INFO => "[Info]",
        MessageSeverity::WARNING => { exit = true; "[Warning]" },
        MessageSeverity::ERROR => { exit = true; "[Error]" },
        _ => "[Unknown]",
    };
    let types = match message_type {
        MessageType::GENERAL => "[General]",
        MessageType::PERFORMANCE => "[Performance]",
        MessageType::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };

    println!("{}{}\n{:?}\n", severity, types, message);
    debug_assert!(!exit, "DAXA DEBUG ASSERTION FAILURE");
}



pub struct ContextInfo {
    pub application_name: Cow<'static, str>,
    pub application_version: u32,
    #[cfg(debug_assertions)] pub validation_callback: ValidationCallback
}

impl Default for ContextInfo {
    fn default() -> Self {
        Self {
            application_name: "Daxa Vulkan App".into(),
            application_version: 0,
            #[cfg(debug_assertions)] validation_callback: default_validation_callback
        }
    }
}



#[derive(Clone)]
pub struct Context(pub(crate) Arc<ContextInternal>);

pub(crate) struct ContextInternal {
    pub entry: Entry,
    pub instance: Instance,
    pub info: Box<ContextInfo>,
    #[cfg(debug_assertions)] _debug_utils: DebugUtils,
    #[cfg(debug_assertions)] _debug_utils_messenger: DebugUtilsMessenger,
}

impl Deref for Context {
    type Target = Instance;

    fn deref(&self) -> &Self::Target {
        &self.0.instance
    }
}

// Context creation methods
impl Context {
    pub fn new(
        info: ContextInfo
    ) -> Result<Self> {
        let entry = ash::Entry::linked();

        let mut info = Box::new(info);

        // Define instance layers and extensions to request
        let layer_names = Vec::from([
            #[cfg(debug_assertions)]
            "VK_LAYER_KHRONOS_validation"
        ]);

        let extension_names: [*const c_char; 3] = [ // TODO: support more surface extensions (+ headless mode?)
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            ash::extensions::khr::Win32Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            ash::extensions::khr::XlibSurface::name().as_ptr(),
        ];

        // Validate support for requested instance layers
        let instance_layers = entry.enumerate_instance_layer_properties()?;
        let validated_layers: Vec<String> = layer_names.iter().map(|&required_layer| {
            debug_assert!(instance_layers.iter().any(|existing_layer| {
                let existing_layer_name = unsafe { CStr::from_ptr(existing_layer.layer_name.as_ptr()).to_str().unwrap() };
                required_layer.eq(existing_layer_name)
            }), "Cannot find layer: {}", required_layer);

            format!("{}\0", required_layer)
        }).collect();
        let validated_layers_raw: Vec<*const c_char> = validated_layers.iter().map(|layer| layer.as_ptr() as *const c_char).collect();

        // Create Vulkan instance
        let application_info = vk::ApplicationInfo::builder()
            .application_name(CString::new(&*info.application_name)?.as_c_str())
            .application_version(info.application_version)
            .engine_name(CString::new("daxa")?.as_c_str())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .build();
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&validated_layers_raw)
            .enabled_extension_names(&extension_names)
            .build();
        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .context("context")?
        };

        // Create debug messenger
        #[cfg(debug_assertions)]
        let (_debug_utils, _debug_utils_messenger) = {
            let user_data = info.as_mut() as *mut ContextInfo as *mut c_void;
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_utils_messenger_callback))
                .user_data(user_data);

            let debug_utils = DebugUtils::new(&entry, &instance);
            let debug_utils_messenger = unsafe {
                debug_utils.create_debug_utils_messenger(&create_info, None)
                    .context("Debug messenger should be created.")?
            };

            (debug_utils, debug_utils_messenger)
        };

        Ok(Self(Arc::new(ContextInternal {
            entry,
            instance,
            info,
            #[cfg(debug_assertions)] _debug_utils,
            #[cfg(debug_assertions)] _debug_utils_messenger,
        })))
    }
}

// Context usage methods
impl Context {
    pub fn create_device(
        &self,
        device_info: DeviceInfo
    ) -> Result<Device> {
        // Get physical devices
        let physical_devices = unsafe {
            self.enumerate_physical_devices()
                .context("Physical devices should be enumerated.")?
        };

        // Score physical devices with provided selector
        let device_score = |physical_device: &vk::PhysicalDevice| -> i32 {
            let device_properties = unsafe { self.get_physical_device_properties(*physical_device) };

            match device_properties.api_version < vk::API_VERSION_1_3 {
                true => 0,
                false => (device_info.selector)(&device_properties)
            }
        };

        let best_physical_device = physical_devices
            .iter()
            .max_by_key(|&a| device_score(a))
            .expect("`physical_devices` should not be empty.");

        debug_assert!(device_score(best_physical_device) != -1, "No suitable device found.");

        // TODO: check every physical device for required features

        // Create logical device from selected physical device
        let physical_device = *best_physical_device;
        let device_properties = unsafe { self.get_physical_device_properties(physical_device) };

        let logical_device = Device::new(device_info, device_properties, self.clone(), physical_device)
                .expect("Device should be created.");

        Ok(logical_device)
    }

    #[cfg(debug_assertions)]
    #[inline]
    pub(crate) fn debug_utils(&self) -> &DebugUtils {
        &self.0._debug_utils
    }
}

impl Drop for ContextInternal {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        unsafe {
            // Safety: vkDestroyDebugUtilsMessengerEXT
            //  Host Synchronization
            //   -  Host access to messenger must be externally synchronized
            //
            //  Messenger is private to this object
            self._debug_utils.destroy_debug_utils_messenger(self._debug_utils_messenger, None);
        }

        unsafe {
            //  Safety: vkDestroyInstance
            //  Host Synchronization
            //   -  Host access to instance must be externally synchronized
            //   -  Host access to all VkPhysicalDevice objects enumerated from instance must be externally synchronized
            //
            //  Synchronized host access to instance guaranteed by borrow checker with '&mut self'
            //  Device objects created with this instance retain a reference, so this should only drop after all Devices drop
            self.instance.destroy_instance(None);
        }
    }
}