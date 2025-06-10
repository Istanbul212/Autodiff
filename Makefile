.PHONY: all build test

all: build

build: tests/autodiff.test.cpp autodiff/graph.h
	@/usr/bin/clang++ -std=gnu++14 -fcolor-diagnostics -fansi-escape-codes -g tests/autodiff.test.cpp -o tests/autodiff.test -std=c++20

test: build
	@tests/autodiff.test
