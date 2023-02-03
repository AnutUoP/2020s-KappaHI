for i in {1..9} #16 for HIHI and zs23, 9 for zs13, 5 for zs8
do
  #python3 F_biasHIHI_7params.py $i
  python3 F_biasKHI_7params.py $i
  #python3 NewF_biasKHI.py $i
done
