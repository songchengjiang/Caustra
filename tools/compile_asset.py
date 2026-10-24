#!/usr/bin/env python3
"""Compile a Blender USD + MaterialX scene to a self-contained .caustra package."""
import argparse
import sys
from pathlib import Path
from caustra_assets.model import AssetError
from caustra_assets.package import write_asset
from caustra_assets.usd_import import import_usd


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="USD/USDA/USDC/USDZ scene exported with MaterialX")
    parser.add_argument("output",type=Path,help="Output .caustra package (embeds all texture data)")
    parser.add_argument("--material",type=Path,help="Explicit standalone .mtlx material applied to all meshes")
    parser.add_argument("--time",type=float,help="USD time-code snapshot (default: authored default values)")
    parser.add_argument("--displacement-subdivisions",type=int,default=3,
                        help="Offline displacement triangle subdivision levels, 0..7 (default: 3; bounded to 1M triangles/mesh)")
    options=parser.parse_args(argv)
    try:
        asset=import_usd(options.input,options.material,options.time,options.displacement_subdivisions)
        write_asset(asset,options.output)
    except (AssetError,OSError) as error:
        print("Caustra asset compiler: "+str(error),file=sys.stderr);return 1
    print("Wrote %s: %d meshes, %d instances, %d MaterialX materials, %d instructions, %d textures" %
          (options.output,len(asset.meshes),len(asset.instances),len(asset.materials),len(asset.instructions),len(asset.textures)))
    return 0


if __name__=="__main__": raise SystemExit(main())
