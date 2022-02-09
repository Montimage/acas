#!/bin/bash

rm -rf libs/

LIBS_DIR=$(pwd)/libs
export MMT_BASE=$LIBS_DIR
export INSTALL_DIR=$LIBS_DIR

mkdir libs

# Compiling SDK
pushd third-party/dpi/sdk/
make -j 8
make install -j 8
popd # pwd

# Compiling Security
pushd third-party/security/
# # In the Makefile, remove line 230 (the copying of rules)
#rm rules/73*
make
make install
popd # pwd/

# # Compiling probe
pushd third-party/probe/
export STATIC_LINK=1
make -j 8
make install -j 8
popd # probe/

cp libs/probe/bin/probe probe
