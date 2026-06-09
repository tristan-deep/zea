"""Lightweight smoke tests for zea.data.app — no HF downloads, no Gradio server."""


def test_build_interface_does_not_crash():
    """build_interface() must construct the Gradio Blocks without raising."""
    from zea.data.app import build_interface

    demo = build_interface()
    # A Gradio Blocks object is truthy and has a .launch method
    assert demo is not None
    assert hasattr(demo, "launch")


def test_zea_app_main_calls_build_interface(monkeypatch):
    """zea.__main__.main() with 'app' dispatches to build_interface and launch."""
    monkeypatch.setattr("sys.argv", ["zea", "app"])

    launched = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    from unittest.mock import patch

    with patch("zea.data.app.build_interface", return_value=_FakeDemo()):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    # share defaults to False, server_port to None
    assert launched.get("share") is False
    assert launched.get("server_port") is None


def test_zea_app_passes_share_flag(monkeypatch):
    """--share flag is forwarded to demo.launch()."""
    monkeypatch.setattr("sys.argv", ["zea", "app", "--share", "--server_port", "7861"])

    launched = {}

    class _FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    from unittest.mock import patch

    with patch("zea.data.app.build_interface", return_value=_FakeDemo()):
        with patch("zea.internal.device.init_device"):
            from zea.__main__ import main

            main()

    assert launched.get("share") is True
    assert launched.get("server_port") == 7861
