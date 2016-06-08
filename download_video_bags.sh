#! /bin/bash
mkdir video_bags
wget http://vmcremers8.informatik.tu-muenchen.de/lsd/LSD_room.bag.zip 
wget http://vmcremers8.informatik.tu-muenchen.de/lsd/LSD_foodcourt.bag.zip
unzip *.zip video_bags
rm LSD_foodcourt.bag.zip
rm LSD_room.ba.zip

