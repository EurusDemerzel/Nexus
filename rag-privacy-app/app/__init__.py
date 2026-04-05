"""Application package.

Expose a canonical Flask app factory without eager side effects.
"""


def create_app(*args, **kwargs):
	"""Lazily import the Flask factory to avoid import-time heavy initialization."""
	from .app import create_app as _create_app

	return _create_app(*args, **kwargs)