#!/bin/bash
domain=${1}
init_val=${2}
sim_directory=${3}
sim_name=${4}
grid_resolution=${5}
support_radius=${6}
method=${7}
sdf_smoothing_factor=${8}
blur_kernel_size=${9}
blur_kernel_offset=${10}
blur_kernel_depth=${11}
blur_surface_cells_only=${12}
blur_iterations=${13}
cff=${14}
mls_similarity_threshold=${15}
mls_max_samples=${16}
app=${17}

for mt in ${method}; do


	if [ ${mt} == "ZhuBridson" ] || [ ${mt} == "ZhuBridsonBlurred" ] || [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "Solenthiler" ]; then
		initVal="-0.5"
	else
		initVal=${init_val}
	fi
	if [ ${mt} == "NaiveMC" ] || [ ${mt} == "NaiveMCBlurred" ] || [ ${mt} == "NaiveMCMls" ]; then
		supportRad="0"
	else 
		supportRad=${support_radius}
	fi
	if [ ${mt} == "ZhuBridsonBlurred" ] || [ ${mt} == "NaiveMCBlurred" ]; then
		sdfsf=${sdf_smoothing_factor}
		bkd=${blur_kernel_depth}
		bit=${blur_iterations}
	else
		sdfsf="1"
		bkd="1"
		bit="1"
	fi

	if [ ${mt} == "ZhuBridsonBlurred" ] || [ ${mt} == "NaiveMCBlurred" ] || [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "NaiveMCMls" ]; then
		bks=${blur_kernel_size}
		bko=${blur_kernel_offset}
		bsfco=${blur_surface_cells_only}
	else
		bks="1"
		bko="1"
		bsfco="true"
	fi
	if [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "NaiveMCMls" ]; then
		msst=${mls_similarity_threshold}
	else
		msst="1"
	fi
	if [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "NaiveMCMls" ]; then
		mlsms=${mls_max_samples}
	else
		mlsms="100"
	fi

	for sfco in ${bsfco}; do

		if [ ${sfco} == "true" ]; then
			lsfco="--blur-surface-cells-only"
		else
			lsfco=""
		fi
		parallel -j 8  \
				./${app} \
					--domain {1} \
					--init-val {2} \
					--sim-name {3} \
					--sim-directory {4} \
					--grid-resolution {5} \
					--support-radius {6} \
					--method {7} \
					--sdf-smoothing-factor {8} \
					--blur-kernel-size {9} \
					--blur-kernel-offset {10} \
					--blur-kernel-depth {11} \
					"${lsfco}" \
					--blur-iterations {12} \
					--cff {13} \
					--mls-similarity-threshold {14} \
					--mls-max-neighbors {15} \
						::: ${domain} ::: ${initVal} ::: ${sim_name}\
						::: ${sim_directory} ::: ${grid_resolution} ::: ${supportRad}\
						::: ${mt} ::: ${sdfsf} ::: ${bks} ::: ${bko}\
						::: ${bkd} ::: ${bit} ::: ${cff} ::: ${msst}\
						::: ${mlsms}
	done
done

