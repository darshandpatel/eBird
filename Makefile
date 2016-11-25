INPUT_FILE="/Users/Darshan/Documents/MapReduce/Data/wikipedia-simple-html.bz2"
HADOOP_ROOT="/usr/local/hadoop"

default:clean
	gradle build
	cp ./build/libs/PageRank.jar .

# No combiner program run
run:
	${HADOOP_ROOT}/bin/hadoop jar PageRank.jar code.Run ${INPUT_FILE} output

# Clean the build
clean:
	rm -rf output .gradle PageRank.jar target build  
