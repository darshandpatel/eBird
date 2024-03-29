INPUT_FILE="/Users/Darshan/Documents/MapReduce/FinalProject/labeled.csv.bz2"
HADOOP_ROOT="/usr/local/hadoop"

default:clean
	gradle build
	cp ./build/libs/eBird.jar .

run:
	${HADOOP_ROOT}/bin/hadoop jar eBird.jar ${INPUT_FILE} output

# Clean the build
clean:
	rm -rf output .gradle eBird.jar target build  
