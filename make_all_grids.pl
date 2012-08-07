#!/usr/bin/perl

#Generate a whole bunch of grids
#
if($ARGV[0] eq 512){
    @nums=(90,141,191,314);
}
else{
    @nums=(91,124,191);
}
@codes=('a','g');
my $outname="omp_submit_script";
foreach my $code (@codes){
foreach my $num (@nums){
    open(my $out, '>', $outname);
print $out
"#!/bin/bash 
#\$ -N dla_$num-$code-512_grid
#\$ -l \"h_rt=16:00:00,exclusive=true\"
#\$ -j y
#\$ -cwd
#\$ -pe orte 4
#\$ -m bae
#\$ -V
export OMP_NUM_THREADS=1
LOCAL=/home/spb/.local
MPI=/usr/local/openmpi/intel/x86_64
FFTW=\$LOCAL/fftw
export PYTHONPATH=\$HOME/.local/python
export LD_LIBRARY_PATH=\${MPI}/lib:\${LD_LIBRARY_PATH}:\$FFTW/lib:/usr/lib64
export LIBRARY_PATH=\${MPI}/lib:\${LIBRARY_PATH}:\$FFTW/lib:/usr/lib64
export PATH=\${MPI}/bin:\$PATH:\$LOCAL/misc/bin

cd \$HOME/codes/ComparisonProject/
python make_a_grid.py $code $num $ARGV[0] 

";
close($out);
`qsub $outname`;
}
}
