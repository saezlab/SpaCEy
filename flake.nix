{
  description = "SpaCEy on NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python311;

        pythonEnv = python.withPackages (ps: with ps; [
          pip
          virtualenv
          setuptools
          wheel

          # CUDA-enabled PyTorch stack from nixpkgs
          torch
          torchvision
          torchaudio

          # Useful base packages
          numpy
          pandas
          scipy
          matplotlib
          jupyterlab
          ipykernel
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.git
            pkgs.pkg-config
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]}:$LD_LIBRARY_PATH

            export VENV_DIR=.venv

            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating virtualenv in $VENV_DIR"
              python -m venv "$VENV_DIR"
            fi

            source "$VENV_DIR/bin/activate"

            export PYTHONNOUSERSITE=1

            # Install your pinned additions once, only if missing
            if ! python -c "import scanpy, pytorch_lightning, leidenalg" >/dev/null 2>&1; then
              pip install --upgrade pip setuptools wheel
              pip install \
                scanpy==1.9.8 \
                pytorch-lightning==2.1.2 \
                leidenalg==0.10.2
            fi

            echo "Python: $(python --version)"
            echo "Torch CUDA available?"
            python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
PY
          '';
        };
      });
}