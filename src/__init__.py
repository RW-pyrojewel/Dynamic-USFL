from __future__ import annotations

# src package 初始化

__all__ = ["__version__", "get_version"]

def _resolve_version() -> str:
    """
    尝试从已安装的分发信息中读取版本号；失败时返回 "0.0.0"。
    适配 importlib.metadata（现代 Python）和 pkg_resources（较老环境）。
    """
    try:
        try:
            from importlib.metadata import version as _md_version  # type: ignore
        except Exception:
            _md_version = None

        if _md_version:
            try:
                return _md_version(__name__)
            except Exception:
                # 包可能未安装为分发包
                pass

        try:
            import pkg_resources  # type: ignore
            return pkg_resources.get_distribution(__name__).version
        except Exception:
            return "0.0.0"
    except Exception:
        return "0.0.0"

__version__ = _resolve_version()
del _resolve_version

def get_version() -> str:
    """返回包的版本号（字符串）。"""
    return __version__