#!/usr/bin/env bash
# train_basic 9000
# for i in {1..2}; do python run.py -w 5 -f 64 -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/basic; done;

# train_skew 9000
for i in {1..4500}; do python run.py -w 5 -f 64 -k 5 -rk -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/skew; done;
for i in {1..4500}; do python run.py -w 5 -f 64 -k 15 -rk -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/skew; done;

# val_distortion 3000
for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 0 -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/dist; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 1 -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/dist; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 2 -l ko --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/dist; done;

# val_blur 3000
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 1 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/blur; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 2 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/blur; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 4 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/blur; done;

# val_background 3000
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 0 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/back; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 1 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/back; done;
for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 2 --output_dir F:/Team_Project/OCR/01_Text_detection/data2_parksunwoo/back; done;


# for i in {1..9000}; do python run.py -w 5 -f 64 -l ko --output_dir asd; done;

# # train_skew 9000
# for i in {1..4500}; do python run.py -w 5 -f 64 -k 5 -rk -l ko --output_dir asd; done;
# for i in {1..4500}; do python run.py -w 5 -f 64 -k 15 -rk -l ko --output_dir asd; done;

# # val_distortion 3000
# for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 0 -l ko --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 1 -l ko --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -d 3 -do 2 -l ko --output_dir asd; done;

# # val_blur 3000
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 1 --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 2 --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -bl 4 --output_dir asd; done;

# # val_background 3000
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 0 --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 1 --output_dir asd; done;
# for i in {1..1000}; do python run.py -w 5 -f 64 -l ko -b 2 --output_dir asd; done;