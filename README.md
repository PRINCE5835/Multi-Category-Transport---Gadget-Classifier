# Multi-Category Transport & Gadget Classifier

## Dataset
Search Kaggle for "vehicles and gadgets image dataset" or "cars bikes phones dataset". Download the zip, extract it into /dataset so you get:
/dataset/Cars/   (put car images here)
/dataset/Bikes/  (put bike images here)
/dataset/Phones/ (put phone images here)

## How to run
cd backend
pip install -r requirements.txt
cd ../model
python train.py
cd ../backend
python app.py
Then open frontend/index.html in a browser and upload an image.
