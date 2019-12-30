file_with_filenames=$1
output_folder=$2
cat $file_with_filenames | while read file
do
	folder=$(echo "$file" | rev | cut -f2,3,4 -d/ | rev)
	echo $folder
	mkdir -p $output_folder/"$folder"
	cp "$file".bvh $output_folder/"$folder"
done
