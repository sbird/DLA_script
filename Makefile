#Python include path
PYINC=-I/usr/include/python2.6 -I/usr/include/python2.6

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
ifeq (icc,$(findstring icc,${CC}))
  CFLAGS +=-O2 -g -c -w1 -openmp -fpic
  LINK +=${CXX} -openmp
else
  CFLAGS +=-O2 -g -c -Wall -fopenmp -fPIC
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
.PHONY: all clean

all: _fieldize_priv.so

clean: _fieldize_priv.so
	rm *.o $^

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $^ -o $@

_fieldize_priv.so: py_fieldize.o SPH_fieldize.o
	$(LINK) $(LFLAGS) -shared $^ -o $@
