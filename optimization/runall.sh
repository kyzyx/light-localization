for f in data/*.lt
do
    echo "./optimization ${f} >> $1"
    ./optimization ${f} >> $1
done
