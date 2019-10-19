BEGIN{
	printf("\"Number of points\",");
	printf("\"CompactSupportNS\",");
	printf("\"brutForceNS\"\n");
	i = 0;
}

/CompactNSearch/ {
	compactNS_ms[i] = $3
	points[i] = $6
}

/custom_nbs/ {
	bfNS_ms[i] = $3
	points[i] = $6
	i++
}

END{
	for(k = 0; k < i; k++){
		printf("\"%d\", \"%d\", \"%d\"\n", points[k], compactNS_ms[k], bfNS_ms[k]);
	}
}