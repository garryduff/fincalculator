{
  pkgs }: {
  description = "Python environment for financial calculators";
  deps = pkgs: with pkgs; [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.flask
    pkgs.python310Packages.flask-cors
    pkgs.python310Packages.numpy
    pkgs.python310Packages.scipy
  ];
}
