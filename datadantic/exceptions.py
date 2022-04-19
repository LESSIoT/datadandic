class ModelError(Exception):
    """Generic model error class."""

    pass


class InvalidDocumentID(ModelError):
    pass


class ModelNotFoundError(ModelError):
    pass
