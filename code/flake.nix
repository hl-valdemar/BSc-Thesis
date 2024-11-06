{
  description = "Flake for my BSc thesis dev shell";
  
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python3
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          # select Python packages here
          numpy
          torch
          matplotlib
        ]))
      ];

      shellHook = ''
        echo "Entered BSc thesis shell."
      '';
    };
  };
}
