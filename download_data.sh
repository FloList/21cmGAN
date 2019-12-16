#!/bin/bash

folder="https://21ssd.obspm.fr/browse/21ssd///lightcones/"
# URL format: https://21ssd.obspm.fr/browse/21ssd///lightcones/fx=0.1_RHS=0.5_fa=0.5_x_lightcone_dtb_3minres.dat  (3 or 6 minres)

declare -a fx=('0.1' '0.3' '1' '3' '10')
declare -a RHS=('0' '0.5' '1')
declare -a fa=('0.5' '1' '2')
declare -a coord=('x' 'y' 'z')

folder_save="/home/flo/PycharmProjects/21cm/Data/low_res"

cd ${folder_save}

for ((i = 0; i < ${#fx[@]}; i++))
do
	for ((j = 0; j < ${#RHS[@]}; j++))
	do
		for ((k = 0; k < ${#fa[@]}; k++))
		do
			for ((l = 0; l < ${#coord[@]}; l++))
			do
				filename=${folder}fx=${fx[$i]}_RHS=${RHS[$j]}_fa=${fa[$k]}_${coord[$l]}_lightcone_dtb_6minres.dat
				echo i=$i: fx=${fx[$i]}, j=$j: RHS=${RHS[$j]}, k=$k: fa=${fa[$k]}, coord=${coord[$l]}.
				wget ${filename} --no-check-certificate
			    echo "Downloaded ${filename}."	
			done
		done
	done
done
echo "Done. All files downloaded."
			
				

