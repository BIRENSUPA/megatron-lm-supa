#!/bin/bash
#===============================================================================
### build.sh - build Biren SUPA Collection release package
### Usage Example:
### bash build.sh -c -a br100 -o whl -f true
### For CD package(CI also call CD pipeline)
### For dev local build
### Options:
###   -h       |  --help                        Print help.
###   -c       |  --clean                       Clean build dir. Only use in local build.
###   -i       |  --internal                    Internal package release.
###   -e       |  --external                    External package release.
###   -v       |  --verbose                     Print complete command. Only for dev.
###   -a <arg> |  --arch <arg>                  Supa gpu arch of BIREN device, build with br100 if not specified.
###   -b <arg> |  --build-type <arg>            <Debug|Release>.
###   -t <arg> |  --build-test <arg>            Enable build testing binary, regression used nightly test , and sanity used for MR CI test.
###   -j <arg> |  --jobs=<arg>                  Parallel compile jobs number.
###   --enable-coverage                         Enable code coverage.
###   --disable-ccache                          Disable ccache.
###   --setup-env                               Setup development environment.
#===============================================================================
# run for every case
set -e

ARCH="arch_20"
SETUP_ENV=OFF
CUR_DIR="$( cd "$( dirname ${BASH_SOURCE[0]} )" >/dev/null 2>&1 && pwd )"
BUILD_DIR="${CUR_DIR}/build"
echo "[INFO] Current directory: ${CUR_DIR}"

while [ $# -gt 0 ]; do
    case "${1}" in
    -h|--help)
        sed -rn 's/^### ?//;T;p;' "$0"
        exit 0
        ;;
    -c|--clean)
        rm -rf build dist *egg-info
        shift
        ;;
    -a|--arch)
      echo -e "[INFO] SUPA ARCH VERSION: $2. ignored. only support 'arch_20'"
      shift 2
      ;;
    -i|--internal)
      for_internal=true
      shift
      ;;
    -e|--external)
      for_external=true
      shift
      ;;
    -t|--build-test)
      enable_build_test="true"
      shift 2
      ;;
    -b|--build-type)
      build_type=$2
      shift 2
      ;;
    --enable-coverage)
      enable_converage=ON
      shift
      ;;
    --setup-env)
      SETUP_ENV=ON
      echo -e "[INFO] Setting up environment only. no more actions."
      shift
      ;;
    -o|--output)
      case "$2" in
        "whl")
          output_type="bdist_wheel"
          shift 2
          ;;
        "tar")
          output_type="bdist"
          shift 2
          ;;
        *)
          echo "unsupported output type '$2'";
          exit 1
        esac
        ;;
    -f|--format)
      case "$2" in
        "true")
          format_arg="true"
          shift 2
          ;;
        "false")
          format_arg="false"
          shift 2
          ;;
        *)
          echo "unsupported format_arg type '$2'";
          exit 1
        esac
        ;;
    --)
        shift
        break
        ;;
    *)
      echo -e "\033[31m[Failed]\033[0m Invalid input params!"
      echo -e "Please use 'build.sh -h' to print acceptable params"
      exit 1
      ;;
  esac
done

export supa_arch=${ARCH}

# install deps which build needs
source "${PWD}"/.env_setup/deps_install.sh
install_uninstall_torch_torchvision install

# update cmake version to override that prebuilt by apt/yum installed
python3 -m pip -q install cmake==3.26.0 || sudo python3 -m pip -q install cmake==3.26.0
export PATH="${HOME}/.local/bin:${PATH}"


# build package.
python3 setup.py bdist_wheel --dist-dir ${CUR_DIR}/dist

# package test files for CI
pkg_name=megatron_supa-test
pkg_dir=${CUR_DIR}/${pkg_name}
[ -d ${pkg_dir} ] && rm -rf ${pkg_dir}
mkdir -p ${pkg_dir}
cp -af ${CUR_DIR}/requirements.txt ${pkg_dir}
cp -af ${CUR_DIR}/pytest.ini ${pkg_dir}
cp -af ${CUR_DIR}/.coveragerc ${pkg_dir}
cp -raf ${CUR_DIR}/.env_setup ${pkg_dir}/env_setup
cp -raf ${CUR_DIR}/tests ${pkg_dir}
cp -raf ${CUR_DIR}/third-party ${pkg_dir}
tar -zcf ${pkg_name}.tar.gz ${pkg_name}
