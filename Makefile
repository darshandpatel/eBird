INPUT_FILE="labeled.csv.bz2"

default:clean
	gradle build
	cp ./build/libs/project.jar .

run:
	hadoop jar project.jar ${INPUT_FILE} output

# Clean the build
clean:
	rm -rf output .gradle project.jar target build  
