# Qdrant text2image search

A basic search by text and by image using CLIP:

* Embed dataset of images

* Save embeddings inside vector db (Qdrant)

* Embed input (image/text) query 

* Search for similar images to input query

* Display them using Streamlit UI

## Setting-up project (hard way)

* Make sure that your OS has python3 and virtual environment installed.

* The whole point of using virtual environment is not to break other dependencie's versions whenever installed in the whole OS. Feel free if you want to take the dependencies from the requirements file and install them on OS without venv.

* To install virtual environment: (I'm using Debian linux)
'''
sudo apt install python3-venv
'''

* Create virtual environment inside the working directory:
'''
python3 -m venv venv
'''

* Activate the virtual environment:
'''
source ./venv/bin/activate
'''

* Install python dependencies from requirements file:
'''
pip3 install -r requirements.txt
'''

* You can explore the "notebooks/txt2img_search.ipynb"

* You can run an UI app using
'''
streamlit run src/app.py
'''

## Setting-up project (easy way)

* Install Docker Engine and Docker Compose:
https://docs.docker.com/engine/install/

* Run below command and the app will start automatically
'''
sudo docker compose run --build streamlit_app
'''

* Click on the first URL (local) when finish building the app:
Network URL: http://172.19.0.2:8501    <-----  (ofcourse it will be different in your case)
External URL: http://41.40.208.130:8501

## Notes

* I've provided a sample images inside data/samples, it contains images of female t-shirts (with different colors)

* If you want to download the whole dataset, download it using the notebook code

## Furthur improvements

* We can use FastAPI to create endpoint for faster and async inferences.