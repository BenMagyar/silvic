//! Links the macOS cdylib with the `TRITONSERVER_*` and `TRITONBACKEND_*` symbols left undefined, as Linux links a
//! shared library by default: the tritonserver process that loads the backend defines them.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
