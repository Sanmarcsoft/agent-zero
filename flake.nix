{
  description = "Agent Zero with MHH-EI Emotional Intelligence integration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      # ── Cross-Compile Law: all images target x86_64-linux ──────────────
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # ── Upstream Agent Zero base image (pinned digest) ─────────────────
      #
      # Agent Zero has 50+ Python deps with native extensions (torch,
      # faiss-cpu, playwright, whisper, etc.) plus Kali system packages,
      # searxng, supervisord, and more. Rather than re-packaging this
      # entire stack in Nix (a multi-week effort), we pin the upstream
      # amd64 image by digest and layer our EI integration on top.
      #
      # This satisfies the Nix Law:
      #   - Image is built via `nix build` with `buildLayeredImage`
      #   - Upstream is pinned (reproducible)
      #   - EI layer is pure Nix
      #
      # To update the upstream pin:
      #   docker manifest inspect agent0ai/agent-zero:latest
      #   nix-prefetch-docker --image-name agent0ai/agent-zero \
      #     --image-digest sha256:<new-digest>
      baseImage = pkgs.dockerTools.pullImage {
        imageName = "agent0ai/agent-zero";
        imageDigest = "sha256:0709b1328e20a7d14fa6eead740f08ad7a69568f37a064b07ca79b9855244f95";
        # Run: nix-prefetch-docker --image-name agent0ai/agent-zero \
        #        --image-digest sha256:0709b1328e20a7d14fa6eead740f08ad7a69568f37a064b07ca79b9855244f95
        # Then paste the sha256 below.
        sha256 = "0000000000000000000000000000000000000000000000000000";  # FIXME: run nix-prefetch-docker
        finalImageName = "agent0ai/agent-zero";
        finalImageTag = "pinned";
        os = "linux";
        arch = "amd64";
      };

      # ── MHH-EI integration layer ──────────────────────────────────────
      #
      # All files that constitute the EI integration, laid out to overlay
      # onto the upstream /a0 (or /git/agent-zero) app directory.

      eiLayer = pkgs.stdenv.mkDerivation {
        name = "agent-zero-ei-layer";
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            let
              relPath = pkgs.lib.removePrefix (toString ./. + "/") (toString path);
              isEI = builtins.any (prefix: pkgs.lib.hasPrefix prefix relPath) [
                # EI Python tools
                "python/tools/self_map_manager.py"
                "python/tools/emotional_analysis.py"
                "python/tools/theory_of_mind.py"
                "python/tools/ei_profile_image.py"
                "python/tools/user_identity_manager.py"
                "python/tools/avatar_generate.py"
                "python/tools/service_health.py"
                # EI helpers and API
                "python/helpers/ei_chromadb_setup.py"
                "python/helpers/ei_backup.py"
                "python/api/ei_metrics.py"
                # EI extensions
                "python/extensions/system_prompt/_30_ei_system_prompt.py"
                "python/extensions/message_loop_prompts_after/_55_ei_emotional_context.py"
                "python/extensions/monologue_end/_80_ei_backup.py"
                # EI prompts
                "prompts/agent.system.tool.self_map_manager.md"
                "prompts/agent.system.tool.emotional_analysis.md"
                "prompts/agent.system.tool.theory_of_mind.md"
                "prompts/agent.system.tool.ei_profile_image.md"
                "prompts/agent.system.tool.user_identity.md"
                "prompts/agent.system.tool.avatar_generate.md"
                "prompts/agent.system.tool.service_health.md"
                # EI agent and knowledge
                "agents/emotional-analyst"
                "lib/mhh-ei/mhh_ei_for_ai_model.md"
                # Config and UI
                "usr/memory/default/behaviour.md"
                "requirements.txt"
                "webui/avatar-test.html"
              ];
            in isEI;
        };
        installPhase = ''
          mkdir -p $out
          # Copy EI files preserving directory structure
          find . -type f | while read f; do
            dir=$(dirname "$f")
            mkdir -p "$out/$dir"
            cp "$f" "$out/$dir/"
          done
        '';
      };

      # ── Startup wrapper that installs chromadb and overlays EI ─────────
      eiStartup = pkgs.writeShellScript "ei-startup.sh" ''
        #!/bin/bash
        set -e

        # Activate the Agent Zero Python venv
        source /opt/venv-a0/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true

        # Install chromadb if not already present (EI dependency)
        python -c "import chromadb" 2>/dev/null || {
          echo "Installing chromadb for EI integration..."
          pip install --quiet "chromadb>=0.4.0" 2>/dev/null || true
        }

        # Overlay EI files into the app directory
        EI_LAYER="${eiLayer}"
        APP_DIR="/a0"

        if [ -d "$EI_LAYER" ]; then
          echo "Applying MHH-EI integration layer..."
          cp -r --no-preserve=ownership "$EI_LAYER"/. "$APP_DIR/" 2>/dev/null || true
        fi

        # Set EI environment
        export EI_CHROMADB_HOST="''${EI_CHROMADB_HOST:-10.0.0.12}"
        export EI_CHROMADB_PORT="''${EI_CHROMADB_PORT:-18000}"

        echo "MHH-EI Emotional Intelligence integration active."
      '';

      # ── OCI Image: upstream base + EI layer ────────────────────────────
      ociImage = pkgs.dockerTools.buildLayeredImage {
        name = "rg.fr-par.scw.cloud/sanmarcsoft/agent-zero-ei";
        tag = "latest";

        fromImage = baseImage;

        contents = [
          eiLayer
        ];

        # Overlay EI files into the app directory and set up startup hooks
        fakeRootCommands = ''
          # Create EI data directories
          mkdir -p ./var/lib/agent-zero/ei-identity
          mkdir -p ./a0/knowledge/custom/main

          # Copy EI layer files into the Agent Zero app directory (/a0)
          if [ -d ${eiLayer} ]; then
            cp -r --no-preserve=ownership ${eiLayer}/. ./a0/ 2>/dev/null || true
            # Copy MHH-EI knowledge from submodule into the knowledge directory
            if [ -f ${eiLayer}/lib/mhh-ei/mhh_ei_for_ai_model.md ]; then
              mkdir -p ./a0/knowledge/custom/main
              cp ${eiLayer}/lib/mhh-ei/mhh_ei_for_ai_model.md ./a0/knowledge/custom/main/
            fi
          fi

          # Install the EI startup hook
          mkdir -p ./ei
          cp ${eiStartup} ./ei/startup.sh
          chmod +x ./ei/startup.sh

          # Patch the Agent Zero run script to source our EI startup
          if [ -f ./exe/run_A0.sh ]; then
            # Prepend EI startup to the existing run script
            sed -i '2i\# MHH-EI integration startup\nsource /ei/startup.sh 2>/dev/null || true\n' ./exe/run_A0.sh 2>/dev/null || true
          fi
        '';

        config = {
          Env = [
            "EI_CHROMADB_HOST=10.0.0.12"
            "EI_CHROMADB_PORT=18000"
            "EI_IDENTITY_REPO_PATH=/var/lib/agent-zero/ei-identity"
          ];
          ExposedPorts = {
            "80/tcp" = {};
          };
          Volumes = {
            "/a0/usr" = {};
            "/var/lib/agent-zero/ei-identity" = {};
          };
        };
      };

    in {
      packages.${system} = {
        # Primary build target: nix build .#packages.x86_64-linux.oci-image
        oci-image = ociImage;

        # EI layer only (for inspection / testing)
        ei-layer = eiLayer;

        default = ociImage;
      };

      # Development shell
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          python312.pkgs.pip
          python312.pkgs.virtualenv
          uv
          nodejs
          ffmpeg-full
          git
          skopeo
          nix-prefetch-docker
        ];
        shellHook = ''
          echo "Agent Zero EI development shell"
          echo ""
          echo "Build OCI image:  nix build .#packages.x86_64-linux.oci-image"
          echo "Push to registry: skopeo copy docker-archive:result docker://rg.fr-par.scw.cloud/sanmarcsoft/agent-zero-ei:latest"
        '';
      };
    };
}
