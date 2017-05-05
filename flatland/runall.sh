for f in ../optimization/data/*.lt
do
    echo "./visual -i $f --print-success -q -r 512 > ${f%.lt}.txt"
    ./visual -i $f --print-success -q -r 512 > ${f%.lt}.txt
done
