use anyhow::{Result, Context as _};

use ash::{
    extensions::ext::DebugUtils,
    Instance,
    vk::{
        self,
        DebugUtilsMessageSeverityFlagsEXT as MessageSeverity,
        DebugUtilsMessageTypeFlagsEXT as MessageType,
        DebugUtilsMessengerCallbackDataEXT as MessageData,
        DebugUtilsMessengerEXT as DebugUtilsMessenger
    },
};

use std::{
    ffi::{CStr, CString},
    ops::Deref,
    os::raw::{c_char, c_void},
    sync::Arc, 
};


#[cfg(debug_assertions)]
type ValidationCallback = unsafe extern "system" fn(MessageSeverity, MessageType, *const MessageData, *mut c_void) -> vk::Bool32;

#[cfg(debug_assertions)]
#[inline]
unsafe extern "system" fn default_validation_callback(
    message_severity: MessageSeverity,
    message_type: MessageType,
    p_message_data: *const MessageData,
    _p_user_data: *mut c_void
) -> vk::Bool32 {
    #[cfg(debug_assertions)]
    {
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
        let message = CStr::from_ptr((*p_message_data).p_message);

        println!("{}{}\n{:?}\n", severity, types, message);
        debug_assert!(exit, "DAXA DEBUG ASSERTION FAILURE");

        vk::FALSE
    }
}

pub struct ContextInfo {
    application_name: &'static str,
    application_version: u32,
    #[cfg(debug_assertions)] validation_callback: ValidationCallback
}

impl Default for ContextInfo {
    fn default() -> Self {
        ContextInfo {
            application_name: "Daxa Vulkan App",
            application_version: 0,
            #[cfg(debug_assertions)] validation_callback: default_validation_callback
        }
    }
}



struct ContextInternal {
    instance: Instance,
    #[cfg(debug_assertions)] _debug_utils: DebugUtils,
    #[cfg(debug_assertions)] _debug_utils_messenger: DebugUtilsMessenger,
    #[cfg(debug_assertions)] _enable_debug_names: bool
}

pub struct Context {
    internal: Arc<ContextInternal>
}

impl Deref for Context {
    type Target = Instance;

    fn deref(&self) -> &Self::Target {
        &self.internal.instance
    }
}

impl Context {
    pub fn new(
        info: ContextInfo
    ) -> Result<Self> {
        let entry = ash::Entry::linked();

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
            .application_name(CString::new(info.application_name)?.as_c_str())
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
                .pfn_user_callback(Some(info.validation_callback))
                .build();

            let debug_utils = DebugUtils::new(&entry, &instance);
            let debug_utils_messenger = unsafe {
                debug_utils.create_debug_utils_messenger(&create_info, None)
                    .context("Debug messenger should be created.")?
            };

            (debug_utils, debug_utils_messenger)
        };

        Ok(Self {
            internal: Arc::new(ContextInternal {
                instance,
                #[cfg(debug_assertions)] _debug_utils,
                #[cfg(debug_assertions)] _debug_utils_messenger,
                #[cfg(debug_assertions)] _enable_debug_names: true
            })
        })
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



#[cfg(test)]
mod tests {
    use super::{Context, ContextInfo, MessageSeverity, MessageType, MessageData};

    #[test]
    fn simplest() {
        let _daxa_context = Context::new(ContextInfo::default())
            .expect("Context should be created.");
    }

    #[test]
    fn custom_validation_callback() {
        unsafe extern "system" fn validation_callback(
            _message_severity: MessageSeverity,
            _message_type: MessageType,
            p_message_data: *const MessageData,
            _p_user_data: *mut std::ffi::c_void
        ) -> ash::vk::Bool32 {
            let message = std::ffi::CStr::from_ptr((*p_message_data).p_message);
        
            println!("{:?}\n", message);
        
            ash::vk::FALSE
        }

        let _daxa_context = Context::new(ContextInfo {
            validation_callback,
            ..Default::default()
        });
    }
}
