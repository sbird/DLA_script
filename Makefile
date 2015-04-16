#Python include path
PYINC=`pkg-config --cflags python2 `
#Comment to enable Kahan summation
OPTS = -DNO_KAHAN
# CC = /usr/bin/gcc
# CXX = /usr/bin/g++
GCCV:=$(shell gcc --version)
ifeq (4.8,$(findstring 4.8,${GCCV}))
	CC = gcc
	CXX = g++
endif
ifeq ($(CC),cc)
  ICC:=$(shell which icc --tty-only 2>&1)
  #Can we find icc?
  ifeq (/icc,$(findstring /icc,${ICC}))
     CC = icc -vec_report0
     CXX = icpc
  else
     GCC:=$(shell which gcc --tty-only 2>&1)
     #Can we find gcc?
     ifeq (/gcc,$(findstring /gcc,${GCC}))
        CC = gcc
        CXX = g++
     endif
  endif
endif

#Are we using gcc or icc?
ifeq (icpc,$(findstring icpc,${CXX}))
  CFLAGS +=-O2 -g -c -w1 -openmp -fpic -DTOP_HAT_KERNEL $(OPTS)
  LINK +=${CXX} -openmp
else
  CFLAGS +=-O3 -g -c -Wall -fopenmp -fPIC -ffast-math -DTOP_HAT_KERNEL $(OPTS)
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
.PHONY: all clean

all: _fieldize_priv.so

clean:
	rm *.o _fieldize_priv.so

%.o: %.cpp fieldize.h
	$(CXX) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $< -o $@

_fieldize_priv.so: py_fieldize.o SPH_fieldize.o
	$(LINK) $(LFLAGS) -shared $^ -o $@
