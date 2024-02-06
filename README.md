# Qdrant text2image search

A basic search by text and by image using CLIP:

* Embed dataset of images

* Save embeddings inside vector db (Qdrant)
* Embed input (image/text) query 
* Search for similar images to input query
* Display them using Streamlit UI

## About dataset


* Zalando dataset is primary used for a task called VITON (virtual try-on) where it contain images of masked shirts, shirts, and poses of females wearing the shirts.
* I've used the cloth (shirts) data for the dataset.
* Link: https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip
* Paper: https://paperswithcode.com/dataset/viton-hd <---- (not related to our task)
  
## Setting-up project (hard way)

* Make sure that your OS has python3 and virtual environment installed.
* The whole point of using virtual environment is not to break other dependencie's versions whenever installed in the whole OS. Feel free if you want to take the dependencies from the requirements file and install them on OS without venv.
* To install virtual environment: (I'm using Debian linux)
```
sudo apt install python3-venv
```
* Create virtual environment inside the working directory:
```
python3 -m venv venv
```
* Activate the virtual environment:
```
source ./venv/bin/activate
```
* Install python dependencies from requirements file:
```
pip3 install -r requirements.txt
```
* You can explore the "notebooks/txt2img_search.ipynb"
* You can run an UI app using
```
streamlit run src/app.py
```

## Setting-up project (easy way)

* Install Docker Engine and Docker Compose:
https://docs.docker.com/engine/install/
* Run below command and the app will start automatically
```
sudo docker compose run --build streamlit_app
```
* Click on the first URL (local) when finish building the app:
* 
Network URL: http://172.19.0.2:8501 <-----(of course it will be different in your case)

External URL: http://41.40.208.130:8501

## Notes

* I've provided a sample images inside data/samples, it contains images of female shirts (with different colors, and shapes)
* If you want to download the whole dataset, download it using the notebook code
* Currently, the dataset that I'm using while embedding is in "data/samples/", you can change it inside "src/app.py"  line 57

## Furthur improvements

* We can use FastAPI to create endpoint for faster and async inferences.
* CLIP is based on zero-shot, if we want something prominent for shirt search, we might fine-tune it first to be more domain specific.
