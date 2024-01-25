import os
import gdown
import platform
import sys

# Check the operating system
if platform.system() == 'Windows':
    print("This script cannot be run on Windows, due to its use of Unix commands..")
    sys.exit(1)

# Download the MS COCO example images from Google Drive. 
if not os.path.exists('../data'):
    id = "1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK"
    output = "../data.tar.gz"
    # Download the tar file, which corresponds to all the files in Gentle Intro, 
    # which contains more than we need.
    gdown.download(id = id, output = output)
    # Unpack and delete the tar file.
    os.system('tar -xf ../data.tar.gz -C ../')
    os.system('rm ../data.tar.gz')
    # Pull out the images and labels
    os.system('mv ../data/coco/examples ../data/coco_image_examples')
    os.system('mv ../data/coco/human_readable_labels.npy ../data/coco_human_readable_labels.npy')
    # Also delete all folders that are not MS-COCO, and rename the remaining folder
    for directory in os.listdir('../data'):
        if directory != 'coco_image_examples' and directory != 'coco_human_readable_labels.npy':
            os.system('rm -r ../data/' + directory)

# Download the 50% split of the train-val set of MS COCO we used. 
id = '1hO2qdHilPe-2HVCCt_UMd1oFFj_qdauU'
output = "../data/cache.tar"
gdown.download(id = id, output = output)
os.system('tar -xf ../data/cache.tar -C ../data')
os.system('rm ../data/cache.tar')
os.system('mv ../data/cache ../data/coco_epochs')

