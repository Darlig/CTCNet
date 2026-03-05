ffmpeg -i ./test_videos/interview.mp4 -filter:v fps=fps=25 ./test_videos/interview25fps.mp4
mv ./test_videos/interview25fps.mp4 ./test_videos/interview.mp4
python ./utils/detectFaces.py --video_input_path ./test_videos/interview.mp4 --output_path ./test_videos/interview/ --number_of_speakers 2 --scalar_face_detection 1.5 --detect_every_N_frame 8
ffmpeg -i ./test_videos/interview.mp4 -vn -ar 16000 -ac 1 -ab 192k -f wav ./test_videos/interview/interview.wav
python ./utils/crop_mouth_from_video.py --video-direc ./test_videos/interview/faces/ --landmark-direc ./test_videos/interview/landmark/ --save-direc ./test_videos/interview/mouthroi/ --convert-gray --filename-path ./test_videos/interview/filename_input/interview.csv
