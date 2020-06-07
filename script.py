import zipfile
from distutils.dir_util import copy_tree
import os
import shutil
import glob


# Name all the video "video'i'" / i 0 -> n
# Add all the videos to directory Videos
# Loop through the directory Videos with each fileName saved in videoTitle (video + i)

def zipdir(path, ziph):
    length = len(path)
    for root, dirs, files in os.walk(path):
        folder = root[length:]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))


videos = [i.split(os.path.sep)[1] for i in glob.glob('Files/Videos/*')]
for video in videos:
    # Name of the video
    videoTitle, videoExtension = video.split(".")
    # Create a copy of the PPTX template
    copy_tree("Files/template",
              "Files/" + videoTitle)

    # Copy the video to /media in the temporary template
    shutil.copy("Files/Videos/" + videoTitle + "." + videoExtension, "Files/" + videoTitle + "/ppt/media")

    # Rename it to media1.mp4
    os.rename("Files/" + videoTitle + "/ppt/media/" + videoTitle + "." + videoExtension, "Files/" + videoTitle + "/ppt/media/media1.mp4")
    # Zip the temp template
    zipf = zipfile.ZipFile('Files/Output/' + videoTitle + '.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('Files/' + videoTitle, zipf)
    zipf.close()
    # Rename the zip to PPTX file
    os.rename("Files/Output/" + videoTitle + ".zip", "Files/Output/" + videoTitle + ".pptx")

    # Delete the temp template
    shutil.rmtree("Files/" + videoTitle)

    # Use convert.py to convert videoTitle.pptx to a Video, save it in Output