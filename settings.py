app = dict(
	# change input and output directories before starting
    IMAGES_INPUT_FOLDER = 'gs://bucket_name/image_folder', 
    OUTPUT_FILENAME = 'gs://bucket_name/tfrecords_out_folder',
    NUMBER_OF_SHARDS = 1, # number of files to split tfrecords into
    TRAINING_EXAMPLES_SPLIT = 0.8,
    SEED = 123
)