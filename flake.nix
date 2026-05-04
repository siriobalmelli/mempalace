{
  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      pyproject-build-systems,
      pyproject-nix,
      uv2nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          python = pkgs.python312;

          workspace = uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          pyprojectOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          pyprojectOverrides = final: prev: {
            chromadb = prev.chromadb.overrideAttrs (old: {
              buildInputs =
                (old.buildInputs or [ ])
                ++ lib.optionals final.stdenv.isLinux [
                  pkgs.stdenv.cc.cc.lib
                ];
              nativeBuildInputs =
                (old.nativeBuildInputs or [ ])
                ++ lib.optionals final.stdenv.isLinux [
                  pkgs.autoPatchelfHook
                ];
            });

            onnxruntime = prev.onnxruntime.overrideAttrs (old: {
              buildInputs =
                (old.buildInputs or [ ])
                ++ lib.optionals final.stdenv.isLinux [
                  pkgs.stdenv.cc.cc.lib
                ];
              nativeBuildInputs =
                (old.nativeBuildInputs or [ ])
                ++ lib.optionals final.stdenv.isLinux [
                  pkgs.autoPatchelfHook
                ];
            });
          };

          pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              pyprojectOverlay
              pyprojectOverrides
            ]
          );

          editableOverlay = workspace.mkEditablePyprojectOverlay {
            root = "$REPO_ROOT";
          };

          editablePythonSet = pythonSet.overrideScope (
            lib.composeManyExtensions [
              editableOverlay
              (final: prev: {
                mempalace = prev.mempalace.overrideAttrs (old: {
                  nativeBuildInputs =
                    (old.nativeBuildInputs or [ ])
                    ++ final.resolveBuildSystem {
                      editables = [ ];
                    };
                });
              })
            ]
          );

          devDependencies = workspace.deps.groups;
        in
        {
          apps.default = {
            meta.description = "Run the mempalace CLI";
            program = "${config.packages.default}/bin/mempalace";
            type = "app";
          };

          checks.coverage =
            pkgs.runCommand "mempalace-coverage"
              {
                nativeBuildInputs = [
                  config.packages.dev
                  pkgs.bash
                  pkgs.git
                ];

                SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              }
              ''
                export HOME="$TMPDIR"
                export PYTHONDONTWRITEBYTECODE=1
                cp -R ${./.} source
                chmod -R u+w source
                cd source
                python -m pytest tests/ -v --ignore=tests/benchmarks --cov=mempalace --cov-report=term-missing --cov-fail-under=85
                touch $out
              '';

          devShells.default = pkgs.mkShell {
            inputsFrom = [ config.packages.dev ];

            packages = [
              pkgs.git
              pkgs.uv
              pythonSet.ruff
            ];

            PYTHONDONTWRITEBYTECODE = "1";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";

            shellHook = ''
              unset PYTHONPATH
            '';
          };

          devShells.editable = pkgs.mkShell {
            inputsFrom = [ config.packages.editable ];

            packages = [
              config.packages.editable
              pkgs.git
              pkgs.uv
            ];

            PYTHONDONTWRITEBYTECODE = "1";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            UV_NO_SYNC = "1";
            UV_PYTHON = editablePythonSet.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT="$(git rev-parse --show-toplevel)"
            '';
          };

          formatter = pkgs.nixfmt;

          packages.default = pythonSet.mkVirtualEnv "mempalace-env" workspace.deps.default;
          packages.dev = pythonSet.mkVirtualEnv "mempalace-dev-env" devDependencies;
          packages.editable = editablePythonSet.mkVirtualEnv "mempalace-editable-env" devDependencies;
          packages.mempalace = config.packages.default;
        };
    };
}
