BEGIN {
	for(i = 0; i < 21; i++){
		stats[i] = 0;
	}
	factor = 100;
}

/(^[0-9]+$)|(^[0-9]*\.[0-9]+$)/{
	val = int($1);
	ind = int(val/factor);

	if(ind >= 20){
		stats[20]++;
	} else {
		stats[ind]++;
	}
}

END {
	for( i = 0; i < 20; i++){
		printf("%d - %d: %d\n", i*factor, (i+1)*factor, stats[i]);
	}
	printf(">%d: %d\n", 20*factor, stats[20]);
}