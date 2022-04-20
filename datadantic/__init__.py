"""
Datadantic
"""
from logging import getLogger
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Type,
    TypeVar,
    cast,
)

import pydantic
from google.cloud import datastore

# there is an async client, which isn't documented... let's stay away from the undocumented parts for now.
# from google.cloud.datastore_v1 import DatastoreClient, DatastoreAsyncClient

from .exceptions import (
    ModelNotFoundError,
)
from . import operators as op


logger = getLogger("datadantic")


# --- configurations
CONFIGURATIONS: Dict[str, Any] = {}


# def configure(db: Union[DatastoreAsyncClient, DatastoreClient], prefix: str = "") -> None:
def configure(db: datastore.Client, namespace: str = "") -> None:
    """
    Configures the prefix and DB

    :param db: The datastore client instance.
    :param prefix: The prefix to use for collection names.
    """
    global CONFIGURATIONS

    CONFIGURATIONS["db"] = db
    CONFIGURATIONS["namespace"] = namespace


def get_client() -> datastore.Client:
    """Get the current client"""
    db = cast(datastore.Client, CONFIGURATIONS.get("db"))
    if db is None:
        raise ValueError("Missing client")
    return db


def get_namespace(db: datastore.Client) -> str:
    """
    Get the current namespace

    Use the user set configuration, fallback to the one configured in the
    client, if any.
    """
    namespace = CONFIGURATIONS.get("namespace")
    if namespace is None:
        namespace = db.namespace
    if namespace is None:
        # Use the default namespace
        namespace = ""
    return namespace


# --- configurations


# --- model
FIND_TYPES = {
    op.LT,
    op.LTE,
    op.EQ,
    op.GT,
    op.GTE,
}


TModel = TypeVar("TModel", bound="Model")


class Model(pydantic.BaseModel):
    """Base model class.

    Implemets basic functionality for Pydantic models, such as save, delete, find, etc.
    """

    __kind__: Optional[str] = None
    __document_id__: str = "id"
    id: Optional[int] = None

    def save(self) -> None:
        """
        Saves this model in the database.
        """
        data = self.dict(by_alias=True)
        if self.__document_id__ in data:
            del data[self.__document_id__]
        key = self._get_doc_key()
        entity = datastore.Entity(
            key, exclude_from_indexes=self._exclude_from_indexes()
        )
        if self.get_document_id() == None:
            self._set_document_id(entity.key.id)
        entity.update(**data)
        db = get_client()
        db.put(entity)

    def delete(self) -> None:
        """
        Deletes this model from the database.
        """
        db = get_client()
        key = self._get_doc_key()
        if key.id is not None:
            db.delete(key)

    def get_document_id(self):
        """
        Get the document ID for this model instance
        """
        doc_id = getattr(self, self.__document_id__, None)
        return doc_id

    @classmethod
    def find(
        cls: Type[TModel],
        filter_: Optional[dict] = None,
        order: Optional[Sequence[str]] = None,
        fetch_params: Optional[Dict[str, Any]] = None,
    ) -> Iterable[TModel]:
        """Returns a list of models from the database based on a filter.

        Example: `Company.find({"company_id": "1234567-8"})`.
        Example: `Product.find({"stock": {">=": 1}})`.

        :param filter_: The filter criteria.
        :param order: Query order.
        :param fetch_params: params send as is to fetch, (limit and the pagers options)
        :return: List of found models.
        """
        if not filter_:
            filter_ = {}
        kwargs = {}
        if order is not None:
            kwargs["order"] = order

        db = get_client()
        namespace = get_namespace(db)
        query = datastore.Query(
            client=db,
            namespace=namespace,
            kind=cls.__kind__,
            **kwargs,
        )

        for key, value in filter_.items():
            query = cls._add_filter(query, key, value)

        if fetch_params is None:
            fetch_params = {}

        doc_iter = query.fetch(**fetch_params)
        pages = doc_iter.pages
        for page in pages:
            for doc in page:
                doc_dict = dict(doc)
                if doc_dict is not None:
                    yield (cls._cls(doc.key.id, doc_dict))

    @classmethod
    def find_one(
        cls: Type[TModel],
        filter_: Optional[dict] = None,
        order: Optional[Sequence[str]] = None,
    ) -> TModel:
        """Returns one model from the DB based on a filter.

        :param filter_: The filter criteria.
        :param order: Query order.
        :raise ModelNotFoundError: If the entry is not found.
        """
        models = cls.find(filter_, order=order, fetch_params={"limit": 1})
        for model in models:
            return model
        raise ModelNotFoundError(f"No '{cls.__name__}' found")

    @classmethod
    def get_by_doc_id(cls: Type[TModel], doc_id: int) -> TModel:
        """Returns a model based on the document ID.

        :param doc_id: The document ID of the entry.
        :return: The model.
        :raise ModelNotFoundError: Raised if no matching document is found.
        """
        db = get_client()
        namespace = get_namespace(db)
        key = db.key(cls.__kind__, doc_id, namespace=namespace)
        entity = db.get(key)
        if entity is None or entity.key is None:
            raise ModelNotFoundError(
                f"No '{cls.__name__}' found with Key({cls.__kind__}, {doc_id}, namespace={namespace})",
            )
        doc_id, doc_dict = entity.key.id, dict(entity)
        return cls._cls(doc_id, doc_dict)

    @classmethod
    def truncate(
        cls: Type[TModel], filter_: Optional[dict], batch_size: int = 128
    ) -> int:
        """Removes documents of the kind matching the given filters

        :param filter_: The filter criteria.
        :param batch_size: Batch size for listing documents.
        :return: Number of removed documents.
        """
        if not filter_:
            filter_ = {}

        db = get_client()
        namespace = get_namespace(db)
        query = datastore.Query(client=db, namespace=namespace, kind=cls.__kind__)
        for key, value in filter_.items():
            query = cls._add_filter(query, key, value)
        query.keys_only()

        count = 0
        while True:
            deleted = 0
            for entity in query.fetch(limit=batch_size):
                db.delete(entity.key)
                deleted += 1
            count += deleted
            if deleted < batch_size:
                return count

    @classmethod
    def _add_filter(
        cls: Type[TModel],
        query: datastore.Query,
        field: str,
        value: Any,
    ) -> datastore.Query:
        "Add filter to the query"
        if type(value) is dict:
            for f_type in value:
                if f_type not in FIND_TYPES:
                    raise ValueError(
                        f"Unsupported filter type: {f_type}. Supported types are : {', '.join(FIND_TYPES)}"
                    )
                query.add_filter(field, f_type, value)
        else:
            query.add_filter(field, op.EQ, value)  # type: ignore
        return query

    @classmethod
    def _cls(cls: Type[TModel], doc_id: Optional[int], data: Dict[str, Any]) -> TModel:
        if cls.__document_id__ in data:
            logger.warning(
                "%s document ID %s contains conflicting %s in data with value %s",
                cls.__name__,
                doc_id,
                cls.__document_id__,
                data[cls.__document_id__],
            )
        data[cls.__document_id__] = doc_id
        model = cls(**data)
        setattr(model, cls.__document_id__, doc_id)
        return model

    def _get_doc_key(self) -> datastore.Key:
        """
        Returns the document reference.
        """
        db = get_client()
        namespace = get_namespace(db)
        doc_id = self.get_document_id()
        if doc_id is not None:
            key = db.key(self.__kind__, doc_id, namespace=namespace)
        else:
            key = db.key(self.__kind__, namespace=namespace)
        return key

    def _set_document_id(self, id: int):
        """
        Set the document ID for this model instance
        """
        return setattr(self, self.__document_id__, id)

    @staticmethod
    def _exclude_from_indexes() -> Sequence[str]:
        """Fields that shouldn't be indexed."""
        return tuple()


# --- model


def list_namespaces() -> Iterator[datastore.Entity]:
    "Helper to lookup namespaces"
    db = get_client()
    query = db.query(kind="__namespace__")
    return query.fetch()


def list_kinds(namespace: str) -> Iterator[datastore.Entity]:
    "Helper to lookup namespaces"
    db = get_client()
    query = db.query(namespace=namespace, kind="__kind__")
    return query.fetch()


__all__ = (
    "Model",
    "configure",
)
