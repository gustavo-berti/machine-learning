import pytest

from pipeline import config

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(
    not (config.DATA_DIR / "imoveis_modelo.csv").exists() or not (config.MODELS_DIR / "regressor.joblib").exists(),
    reason="rode `python -m pipeline` antes",
)


def test_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 5
