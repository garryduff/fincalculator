{
  pkgs }: {
    deps = [
      pkgs.python39
      pkgs.python39Packages.flask
      pkgs.python39Packages.numpy
      pkgs.python39Packages.scipy
    ];
}
