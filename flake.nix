{
  description = "Flake for my BSc thesis dev shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        typst
        python3
        (pkgs.python3.withPackages (python-pkgs:
          with python-pkgs; [
            # select Python packages here
            bpython
            numpy
            torch
            matplotlib
            seaborn
            distutils
            wandb
            pandas
            glob2
          ]))
      ];

      # Virtual env necessary as nix doesn't have the normflows package...
      buildInputs = with pkgs; [
        (python3.withPackages (ps: [
          # nix-managed packages
        ]))
        python3Packages.pip
        python3Packages.virtualenv
      ];

      shellHook = ''
        echo "Entered BSc thesis shell."
      '';
    };
  };
}
