#!/bin/bash

cmake -S src -B build-dev -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-dev