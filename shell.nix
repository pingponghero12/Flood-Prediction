{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = [
    (pkgs.python312.withPackages (
      ps: with ps; [
        rasterio
        numpy
        matplotlib
        requests
        whitebox
      ]
    ))
    pkgs.gdal
  ];
}
