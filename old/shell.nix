# with import <nixpkgs> {};

# let
#   pythonPackages = pkgs.python38Packages;
# in

# pkgs.mkShell {

#     name = "impurePythonEnv";
#     venvDir = "./venv";

#     buildInputs = with pkgs; [
#         pythonPackages.python
#         pythonPackages.venvShellHook
#         pythonPackages.setuptools
        
#         # in order to compile any binary extensions they may
#         # require
#         taglib
#         openssl
#         git
#         libxml2
#         libxslt
#         libzip
#         zlib
#         libGL
#         glib
#     ];

#     postVenvCreation = ''
#         unset SOURCE_DATE_EPOCH
#         pip install -r requirements.txt
#     '';

#     postShellHook = ''
#         # allow pip to install wheels
#         unset SOURCE_DATE_EPOCH
#         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.libGL pkgs.zlib pkgs.glib pkgs.cudatoolkit_10_1 pkgs.cudnn_cudatoolkit_10_1 pkgs.cudatoolkit_10_1.lib]}
#         alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='$HOME' \pip"
#         export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.7/site-packages:$PYTHONPATH"
#         export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
#         unset SOURCE_DATE_EPOCH
#     '';
# }


{ pkgs ? import <nixpkgs> { 
	config.allowUnfree = true;
} }:
let
  venvDir = "./.venv-nix";

  # These are necessary for taichi at runtime.
  libs = [
    pkgs.stdenv.cc.cc.lib
    pkgs.xorg.libX11
    pkgs.ncurses5
    pkgs.linuxPackages.nvidia_x11
    pkgs.libGL
    pkgs.libzip
    pkgs.glib
    pkgs.cudatoolkit_11
    pkgs.cudnn_cudatoolkit_11
  ];
in
pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    # pkgs.python3Packages.tensorflowWithCuda
    # pkgs.cudatoolkit
  ];

  nativeBuildInputs = [ pkgs.cudatoolkit_11 pkgs.cudnn_cudatoolkit_11 ];

  shellHook = ''
    if [ -d "${venvDir}" ]; then
      echo "Virtualenv '${venvDir}' already exists"
    else
      echo "Creating virtualenv '${venvDir}'"
      ${pkgs.python3Packages.python.interpreter} -m venv "${venvDir}"
    fi
    source "${venvDir}/bin/activate"
    pip install -r ./requirements.txt
  '';

  LD_LIBRARY_PATH = "${pkgs.stdenv.lib.makeLibraryPath libs}";
  CUDA_PATH = "${ pkgs.cudatoolkit_11 }";
}
