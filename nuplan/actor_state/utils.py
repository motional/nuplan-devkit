from typing import Any, Callable


def lazy_property(f: Callable[..., Any]) -> Any:
    """ Decorator to make lazy-evaluated properties"""
    attribute = '_' + f.__name__

    @property  # type: ignore
    def _lazy_property(self: object) -> Any:
        if not hasattr(self, attribute):
            setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return _lazy_property
