# Builds libdet_metal.dylib.  The shaders (shaders/*.metal) are compiled at runtime from source by the Metal
# framework, so no Metal toolchain is required (only Xcode / the command line tools for clang + frameworks).
CXX ?= clang++
CXXFLAGS = -std=c++17 -ObjC++ -fobjc-arc -O2 -Wall -Wno-unused-function -ffp-contract=off -fno-fast-math
LDFLAGS = -framework Metal -framework Foundation -dynamiclib
SRCS = det_metal.mm glue.mm gemm.mm attn.mm gdn.mm sample.mm
OBJS = $(SRCS:.mm=.o)
all: libdet_metal.dylib
%.o: %.mm det_metal.h runtime.h params.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
libdet_metal.dylib: $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@
clean:
	rm -f *.o libdet_metal.dylib
