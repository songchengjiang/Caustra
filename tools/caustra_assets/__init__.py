"""Optional USD/MaterialX asset compiler. Nothing here is imported by the renderer."""

from .model import Asset, AssetError, Material, Mesh, Texture
from .package import read_asset, write_asset

__all__ = ["Asset", "AssetError", "Material", "Mesh", "Texture", "read_asset", "write_asset"]
