for f in *.lt
do
    echo "./flatland 300 300 ${f%.lt}.ppm $f -cuda"
    ./flatland 300 300 ${f%.lt}.ppm $f -cuda
done
