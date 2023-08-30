use daxa_rs::{
    context::*,
    device::*
};


const APPNAME: &str = "Daxa API Test: TaskGraph";
const APPNAME_PREFIX: &str = "[Daxa API Test: TaskGraph]";


pub struct AppContext {
    pub context: Context,
    pub device: Device
}

impl AppContext {
    pub fn new() -> Self {
        let context = Context::new(ContextInfo {
            application_name: APPNAME.into(),
            application_version: 1,
            ..Default::default()
        }).unwrap();

        let device = context.create_device(DeviceInfo {
            debug_name: format!("{} device", APPNAME_PREFIX).into(),
            ..Default::default()
        }).unwrap();

        Self {
            context,
            device
        }
    }
}