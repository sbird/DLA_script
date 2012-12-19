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
  CFLAGS +=-O2 -g -c -w1 -openmp -I${GREAD} -fpic -std=gnu99
  LINK +=${CXX} -openmp
else
  CFLAGS +=-O2 -g -c -Wall -fopenmp -I${GREAD} -fPIC -std=gnu99
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
.PHONY: all

all: _fieldize_priv.so

clean:
	rm py_fieldize.o _fieldize_priv.so

py_fieldize.o: py_fieldize.c
	$(CC) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $^ -o $@
_fieldize_priv.so: py_fieldize.o
	$(LINK) -shared $^ -o $@
