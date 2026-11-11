use std::sync::Arc;

/// Every failure silvic reports. Cheap to clone: a collective failure on one rank is surfaced on every call after it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SilvicError {
    #[error("invalid parameter: {0}")]
    Config(Arc<str>),
    #[error("invalid training data: {0}")]
    Data(Arc<str>),
    #[error("device error: {0}")]
    Device(Arc<str>),
    #[error("collective failed: {0}")]
    Collective(Arc<str>),
    #[error("malformed model: {0}")]
    Model(Arc<str>),
    #[error("unsupported: {0}")]
    Unsupported(Arc<str>),
    /// The iteration callback asked to stop; carries the callback's reason.
    #[error("training interrupted: {0}")]
    Interrupted(Arc<str>),
    /// A training invariant failed (a bug, never bad input); training stops instead of panicking.
    #[error("internal invariant violated: {0}")]
    Internal(Arc<str>),
}

impl SilvicError {
    /// Stable label for the Python exception mapping and logs.
    pub fn kind(&self) -> &'static str {
        match self {
            SilvicError::Config(_) => "config",
            SilvicError::Data(_) => "data",
            SilvicError::Device(_) => "device",
            SilvicError::Collective(_) => "collective",
            SilvicError::Model(_) => "model",
            SilvicError::Unsupported(_) => "unsupported",
            SilvicError::Interrupted(_) => "interrupted",
            SilvicError::Internal(_) => "internal",
        }
    }

    pub fn config(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Config(msg.into())
    }

    pub fn data(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Data(msg.into())
    }

    pub fn internal(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Internal(msg.into())
    }

    pub fn device(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Device(msg.into())
    }

    pub fn collective(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Collective(msg.into())
    }

    pub fn model(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Model(msg.into())
    }

    pub fn unsupported(msg: impl Into<Arc<str>>) -> Self {
        SilvicError::Unsupported(msg.into())
    }
}

pub type Result<T, E = SilvicError> = std::result::Result<T, E>;
