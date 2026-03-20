# flake.nix for ML with GPU
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

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
        python = pkgs.python3.withPackages (ps: with ps; [
          torch         # PyTorch with CUDA
          torchvision
          numpy
          pandas
          matplotlib
          jupyter
          tensorboard
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];
          shellHook = ''
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}