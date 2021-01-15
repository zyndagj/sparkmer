#hdfs dfs -mkdir Bacteria
for path in `lfs find Bacteria -name \*.fna`
do
	hdfs dfs -mkdir -p `dirname $path`
	hdfs dfs -put $path $path &
done
