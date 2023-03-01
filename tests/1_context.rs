
use daxa_rs::context::*;

#[test]
fn simplest() {
    let daxa_context = Context::new(ContextInfo::default());

    assert!(daxa_context.is_ok())
}

#[test]
fn custom_validation_callback() {
    fn validation_callback(
        _message_severity: MessageSeverity,
        _message_type: MessageType,
        message: &std::ffi::CStr,
    ) {
        println!("{:?}\n", message);
    }

    let daxa_context = Context::new(ContextInfo {
        validation_callback,
        ..Default::default()
    });

    assert!(daxa_context.is_ok())
}
