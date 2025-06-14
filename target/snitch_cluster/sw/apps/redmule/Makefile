# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yvan Tortorella <yvan.tortorella@unibo.it>
#
# RedMulE Golden Model Makefile

SHELL   := /bin/bash
CUR_DIR := $(shell pwd)
PENV    := $(CUR_DIR)/venv
SW      ?= $(shell pwd)/inc

## Perform Matrix-Matrix Operations of the kins _Z = Y (op2) (X (op1) W)_
## We assume X is _MxN_, W is _NxK_, while Y and Z are _MxK_

# Operation
OP ?= gemm

# Tensors Dimensions
M ?= 32
N ?= 32
K ?= 32

G          ?= 16
QUANT_BITS ?= 8
H          ?= 4
P          ?= 1

# FP format
fp_fmt ?= FP16

check_sw:
	mkdir -p $(SW)

check_penv: check_sw
ifndef PENV
	$(error PENV is undefined. Make sure to export       \
	PENV=<absolute_path_your_pythonenv> before continue.)
	python3 -m venv $(PENV)
	source .venv/bin/activate
	pip3 install numpy
	pip3 install torch
endif

# The current Makefile can be used as follows:
# make $(operation)      -> execute any of the supported GEMM-Ops operation (Z = (X op1 W) op2 Y)
# Set the tensor dimensions while launching the Makefile. For example, in order to run a GEMM
# between three 64x64 matrices, run:
# `make clean gemm M=64 N=64 K=64`

golden: $(OP)

gemm: check_penv
	mkdir -p $@/txt;                          \
	source $(PENV)/bin/activate;              \
	python3 ./$@.py                           \
	--m_size $(M) --n_size $(N) --k_size $(K) \
	--inc_dir $(SW)                           \
	--txt_dir $(CUR_DIR)/$@/txt;              \
	deactivate

gptq_gemm: check_penv
	mkdir -p $@/txt;                          \
	cd $(PENV);                               \
	source ./bin/activate;                    \
	python3 ./$@.py                           \
	--m_size $(M) --n_size $(N) --k_size $(K) \
	--n_groups $(G) --n_bits $(QUANT_BITS)    \
	--height $(H) --pipe_regs $(P)   		  \
	--inc_dir $(SW)                           \
	--txt_dir $(CUR_DIR)/$@/txt;              \
	cd $(PENV);                               \
	deactivate

pace: check_penv
	mkdir -p $@/txt;                          \
	cd $(PENV);                               \
	source ./bin/activate;                    \
	python3 ./$@.py                           \
	--inc_dir $(SW)                           \
	--txt_dir $(CUR_DIR)/$@/txt;              \
	cd $(PENV);                               \
	deactivate

golden-clean:
	rm -rf net_parameters.h
	rm -rf __pycache__
	rm -rf gemm gptq_gemm pace
