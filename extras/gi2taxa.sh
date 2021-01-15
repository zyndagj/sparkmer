while read gi; do
	URL="http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&id=${gi}&rettype=fasta&retmode=xml"
	curl -s $URL | grep "TaxId" | sed -e 's/[[:blank:]]//g' -e 's/<[^>]*>//g'
done < $1
