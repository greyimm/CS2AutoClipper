import cv2
print("OpenCV version:", cv2.__version__)
print("Python and OpenCV are working!")

# USE OPENCV FOR VIDEO PROCESSING AND DETECING THE RED BORDERS
# AND THE GUN SYMBOLS ON THE TOP RIGHT OF THE SCREEN.

# The logic should be something along the lines of: 
# If RedOutlineDetected and GunIconDetected
#   trim beginning of video 5 seconds before they were detected
# (In cases where 5 seconds before they were detected is not available, do not trim at all.)
# Next, find out how many times RedOutlineDetected and GunDetected occurred.
# Rename file based on how many times RedOutlineDetected and GunDetected occurred.
#   export video to TrimmedVideos folder
# elif RedOutlineDetected (and not GunIconDetected)
#   do the same logic as the 'If RedOutlineDetected and GunIconDetected'
#   EXCEPT Rename the file only based on the times RedOutlineDetected
#   And put it into an unsorted folder



# KNOWN ISSUES / UPGRADES :
# Make sure that the program accounts for rectangles moving upwards by more than 120px.  


# Make it so that if the program detects a red outlined rectangle, it will also check within the rectangle for a white gun icon. 
# Make sure to explain how to feed in the exact shape of each icon so that the program will know because
# the icon of the gun is always the exact same. 


#Go through a handbrake compression tutorial video and find out how to get it to auto compress based on the size of the video. 
#There was a YouTuber that created a really useful chart that we should consult in the making of the compressor. 
