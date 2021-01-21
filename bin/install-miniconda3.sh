[ -z "$1" ] && PREFIX="." || PREFIX="$1"
[ -d "$PREFIX/packages/miniconda3" ] || mkdir -p $PREFIX/packages/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$PREFIX/packages/miniconda3/Miniconda3-latest-Linux-x86_64.sh"
chmod +x "$PREFIX/packages/miniconda3/Miniconda3-latest-Linux-x86_64.sh"
$PREFIX/packages/miniconda3/Miniconda3-latest-Linux-x86_64.sh -b -p "$PREFIX/miniconda3"
