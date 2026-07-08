# Multi-Category Transport & Gadget Classifier

## Dataset
Search Kaggle for "vehicles and gadgets image dataset" or "cars bikes phones dataset". Download the zip, extract it into /dataset so you get:
/dataset/Cars/
/dataset/Bikes/
/dataset/Phones/

## How to run locally
cd backend
pip install -r requirements.txt
cd ../model
python train.py
cd ../backend
python app.py
Then open frontend/index.html in a browser.

## Deployed Links
- **Frontend (Vercel):** https://frontend-ten-snowy-7eut4v1ct0.vercel.app
- **Backend API (Render):** https://multi-category-transport-gadget.onrender.com
- **API Endpoint:** POST https://multi-category-transport-gadget.onrender.com/predict
