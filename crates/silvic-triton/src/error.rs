use std::sync::Arc;

use silvic_core::SilvicError;

/// Every failure the backend reports to Triton: model-load errors fail the load, request errors become that request's
/// error response. Cheap to clone: one scoring failure answers every request scored with it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// `config.pbtxt` does not describe a model this backend serves.
    #[error("invalid model configuration: {0}")]
    Config(Arc<str>),
    /// The model file could not be read.
    #[error("cannot read {path}: {message}")]
    Io { path: Arc<str>, message: Arc<str> },
    /// silvic could not parse or compile the model.
    #[error(transparent)]
    Model(#[from] SilvicError),
    /// A request's input does not match the model configuration.
    #[error("invalid request: {0}")]
    Request(Arc<str>),
    #[error("unsupported: {0}")]
    Unsupported(Arc<str>),
    /// A Triton API call failed; `code` is its `TRITONSERVER_Error_Code`.
    #[error("{message}")]
    Server { code: u32, message: Arc<str> },
    /// Code under an exported function panicked; the panic stopped there.
    #[error("silvic backend panicked: {0}")]
    Panic(Arc<str>),
}

impl Error {
    /// Stable label for logs.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Config(_) => "config",
            Error::Io { .. } => "io",
            Error::Model(_) => "model",
            Error::Request(_) => "request",
            Error::Unsupported(_) => "unsupported",
            Error::Server { .. } => "server",
            Error::Panic(_) => "panic",
        }
    }

    pub fn config(msg: impl Into<Arc<str>>) -> Self {
        Error::Config(msg.into())
    }

    pub fn request(msg: impl Into<Arc<str>>) -> Self {
        Error::Request(msg.into())
    }

    pub fn unsupported(msg: impl Into<Arc<str>>) -> Self {
        Error::Unsupported(msg.into())
    }

    /// From a `catch_unwind` payload: the panic message when it is a string.
    pub fn panic(payload: &(dyn std::any::Any + Send)) -> Self {
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("a non-string payload");
        Error::Panic(message.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_payloads_keep_their_message() {
        let caught = std::panic::catch_unwind(|| panic!("row {} out of range", 7)).expect_err("panics");
        assert_eq!(
            Error::panic(&*caught).to_string(),
            "silvic backend panicked: row 7 out of range"
        );
        let caught = std::panic::catch_unwind(|| std::panic::panic_any(3_u8)).expect_err("panics");
        assert_eq!(
            Error::panic(&*caught).to_string(),
            "silvic backend panicked: a non-string payload"
        );
    }

    #[test]
    fn kinds_and_messages() {
        let cases = [
            (Error::config("x"), "config", "invalid model configuration: x"),
            (Error::request("x"), "request", "invalid request: x"),
            (Error::unsupported("x"), "unsupported", "unsupported: x"),
            (Error::from(SilvicError::model("x")), "model", "malformed model: x"),
            (
                Error::Server {
                    code: 3,
                    message: "x".into(),
                },
                "server",
                "x",
            ),
        ];
        for (error, kind, message) in cases {
            assert_eq!(error.kind(), kind, "{message}: kind");
            assert_eq!(error.to_string(), message, "{kind}: message");
        }
    }
}
