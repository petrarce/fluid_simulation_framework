#!/bin/bash

domain="-3,-3,-3,3,3,3"
init_val="-0.5"
sim_directory="./"
sim_name="new_sim"
grid_resolution="0.02"
support_radius="0.08"
method="ZhuBridson"
sdf_smoothing_factor="1"
blur_kernel_size="1"
blur_kernel_offset="1"
blur_kernel_depth="0.5"
blur_surface_cells_only="true"
blur_iterations="1"
cff="0"
mls_max_samples="100"
mls_samples="100"
mls_curvature_particles="20"
mls_overlap_factor="0.5"
mls_cluster_factor="0.5"
app="_empty_command_"
num_threads=8
nested_paralelism="TRUE"
parallel_options="-j8 --ungroup"
tag=""



function printHelp()
{
	echo '-h|--help'
	echo '-d|--domain'
	echo '-i|--init-val'
	echo '-s|--sim-dir'
	echo '-n|--sim-name'
	echo '-g|--grid-res'
	echo '-r|--support-radius'
	echo '-m|--method'
	echo '-f|--sdf-smoothing-factor'
	echo '-ks|--blur-kernel-size'
	echo '-ko|--blur-kernel-offset'
	echo '-kd|--blur-kernel-depth'
	echo '-sco|--blur-surface-cells-only'
	echo '-bi|--blur-iterations'
	echo '--cff'
	echo '-st|--mls-similarity-threshold'
	echo '-ms|--mls-samples'
	echo '-mms|--mls-max-samples'
	echo '-mcf|--mls-clusters-factor'
	echo '-cp | --mls-curvature-particles'
	echo '--app'
	echo '-nt|--num-threads'
	echo '-np|--nested-paralelism'
	echo '--parallel-opts'
	echo '--tag'
}


while [ $# -gt 0 ]; do
	key=$1
	case ${key} in
		-h|--help)
		printHelp
		exit 1
		;;

		-mcf|--mls-clusters-factor)
		if [ $# -lt 2 ]; then exit; fi
		mls_cluster_factor=${2}
		shift
		shift
		;;

		-np|--nested-paralelism)
		if [ $# -lt 2 ]; then exit; fi
		nested_paralelism=${2}
		shift
		shift
		;;

		-ms|--mls-samples)
		if [ $# -lt 2 ]; then exit; fi
		mls_samples=${2}
		shift
		shift
		;;



		--tag)
		if [ $# -lt 2 ]; then exit; fi
		tag=${2}
		shift
		shift
		;;


		--parallel-opts)
		if [ $# -lt 2 ]; then exit; fi
		parallel_options=${2}
		shift
		shift
		;;


		-d|--domain)
		if [ $# -lt 2 ]; then exit; fi
		domain=${2}
		shift
		shift
		;;

		-i|--init-val)
		if [ $# -lt 2 ]; then exit; fi
		init_val=${2}
		shift
		shift
		;;

		-s|--sim-dir)
		if [ $# -lt 2 ]; then exit; fi
		sim_directory=${2}
		shift
		shift
		;;

		-n|--sim-name)
		if [ $# -lt 2 ]; then exit; fi
		sim_name=${2}
		shift
		shift
		;;

		-g|--grid-res)
		if [ $# -lt 2 ]; then exit; fi
		grid_resolution=${2}
		shift
		shift
		;;

		-r|--support-radius)
		if [ $# -lt 2 ]; then exit; fi
		support_radius=${2}
		shift
		shift
		;;

		-m|--method)
		if [ $# -lt 2 ]; then exit; fi
		method=${2}
		shift
		shift
		;;

		-f|--sdf-smoothing-factor)
		if [ $# -lt 2 ]; then exit; fi
		sdf_smoothing_factor=${2}
		shift
		shift
		;;

		-ks|--blur-kernel-size)
		if [ $# -lt 2 ]; then exit; fi
		blur_kernel_size=${2}
		shift
		shift
		;;

		-ko|--blur-kernel-offset)
		if [ $# -lt 2 ]; then exit; fi
		blur_kernel_offset=${2}
		shift
		shift
		;;

		-kd|--blur-kernel-depth)
		if [ $# -lt 2 ]; then exit; fi
		blur_kernel_depth=${2}
		shift
		shift
		;;

		-sco|--blur-surface-cells-only)
		if [ $# -lt 2 ]; then exit; fi
		blur_surface_cells_only=${2}
		shift
		shift
		;;

		-bi|--blur-iterations)
		if [ $# -lt 2 ]; then exit; fi
		blur_iterations=${2}
		shift
		shift
		;;

		--cff)
		if [ $# -lt 2 ]; then exit; fi
		cff=${2}
		shift
		shift
		;;

		-st|--mls-similarity-threshold)
		if [ $# -lt 2 ]; then exit; fi
		mls_similarity_threshold=${2}
		shift
		shift
		;;

		-mms|--mls-max-samples)
		if [ $# -lt 2 ]; then exit; fi
		mls_max_samples=${2}
		shift
		shift
		;;

		-of|--mls-overlap-factor)
		if [ $# -lt 2 ]; then exit; fi
		mls_overlap_factor=${2}
		shift
		shift
		;;

		-cp | --mls-curvature-particles)
		if [ $# -lt 2 ]; then exit; fi
		mls_curvature_particles=${2}
		shift
		shift
		;;

		--app)
		if [ $# -lt 2 ]; then exit; fi
		app=${2}
		shift
		shift
		;;

		-nt|--num-threads)
		if [ $# -lt 2 ]; then exit; fi
		num_threads=${2}
		shift
		shift
		;;

		*)
		echo "UNKNOWN PARAMETER ${key}"
		printHelp
		exit
		;;

	esac
done

echo domain=${domain} init_val=${init_val} sim_directory=${sim_directory}\
	  sim_name=${sim_name} grid_resolution=${grid_resolution} support_radius=${support_radius}\
	  method=${method} sdf_smoothing_factor=${sdf_smoothing_factor}\
	  blur_kernel_size=${blur_kernel_size} blur_kernel_offset=${blur_kernel_offset}\
	  blur_kernel_depth=${blur_kernel_depth} blur_surface_cells_only=${blur_surface_cells_only}\
	  blur_iterations=${blur_iterations} cff=${cff} mls_similarity_threshold=${mls_similarity_threshold}\
	  mls_max_samples=${mls_max_samples} app=${app} tag=${tag}

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

	if [ ${mt} == "ZhuBridsonBlurred" ] || [ ${mt} == "NaiveMCBlurred" ] ; then
		bks=${blur_kernel_size}
		bko=${blur_kernel_offset}
		bsfco=${blur_surface_cells_only}
		bkd=${blur_kernel_depth}
	else
		bks="1"
		bkd="1"
		bko="1"
		bsfco="true"
	fi


	if [ ${mt} == "ZhuBridsonBlurred" ] || [ ${mt} == "NaiveMCBlurred" ] || [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "NaiveMCMls" ]; then
		bit=${blur_iterations}
		sdfsf=${sdf_smoothing_factor}
	else
		sdfsf="1"
		bit="1"
	fi
	if [ ${mt} == "ZhuBridsonMls" ] || [ ${mt} == "NaiveMCMls" ]; then
		mms=${mls_max_samples}
		cp=${mls_curvature_particles}
		of=${mls_overlap_factor}
		ms=${mls_samples}
		mcf=${mls_cluster_factor}
	else
		mms="1"
		cp="1"
		of="1"
		ms="1"
		mcf="1"
	fi

	for sfco in ${bsfco}; do

		if [ ${sfco} == "true" ]; then
			lsfco="--blur-surface-cells-only"
		else
			lsfco=""
		fi
		
		OMP_NUM_THREADS=${num_threads} OMP_NESTED=${nested_paralelism} parallel ${parallel_options} \
				${app} \
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
					--mls-max-samples {14} \
					--mls-curvature-particles {15}\
					--mls-sample-overlap-factor {16}\
					--tag {17}\
					--mls-samples {18}\
					--mls-cluster-fraction {19}\
						::: ${domain} ::: ${initVal} ::: ${sim_name}\
						::: ${sim_directory} ::: ${grid_resolution} ::: ${supportRad}\
						::: ${mt} ::: ${sdfsf} ::: ${bks} ::: ${bko}\
						::: ${bkd} ::: ${bit} ::: ${cff} ::: ${mms} ::: ${cp}\
						::: ${of} ::: ${tag} ::: ${ms} ::: ${mcf}
	done
done

