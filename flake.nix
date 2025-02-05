{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system; 
      };
    in 
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          ruby
          # rubyPackages_3_4.jekyll

          openssl
          pkg-config
          cmake
        ];
      };
    };
}
