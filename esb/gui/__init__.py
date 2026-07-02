"""esb.gui — schema-driven Streamlit front end (design doc §8).

The form is rendered entirely from ``esb.config.describe()``; no field names
are hardcoded here. ``streamlit`` is an OPTIONAL dependency: nothing outside
``app.py`` imports it, so the CLI/pipeline never require it.
"""
