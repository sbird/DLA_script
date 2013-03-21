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
  CFLAGS +=-O2 -g -c -w1 -openmp -fpic -std=gnu99
  LINK +=${CXX} -openmp
else
  CFLAGS +=-O2 -g -c -Wall -fopenmp -fPIC -std=gnu99
  LINK +=${CXX} -openmp $(PRO)
  LFLAGS += -lm -lgomp
endif
.PHONY: all clean

all: _fieldize_priv.so _power_priv.so

clean: _fieldize_priv.so _power_priv.so
	rm *.o $^

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -fno-strict-aliasing -DNDEBUG $(PYINC) -c $^ -o $@

_%_priv.so: py_%.o
	$(LINK) $(LFLAGS) -shared $^ -o $@

_power_priv.so: py_power.o
	$(LINK) $(LFLAGS) -lfftw3 -lfftw3_threads -shared $^ -o $@

