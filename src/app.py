# for UI
import streamlit as st
# list directory using regex
import glob
# for models
from transformers import CLIPProcessor, TFCLIPModel
# for vector db
from qdrant_client import QdrantClient
from qdrant_client.http import models
# different funcs
from utils import download_extract_dataset, embed_dataset
# for basic operations
import numpy as np
# to load images from disk
from PIL import Image

def embed_text(
    text_query:str,
    data_processor,
    embedder
) -> list:
    
    
    # tokenize input text
    processed_text = data_processor(
        text=text_query,
        images=None,
        return_tensors="tf",
        padding=True,
        truncation=True
    )

    # embed tokens
    text_embedding = embedder.get_text_features(**processed_text)

    # convert eager tensor to numpy array and return
    return np.squeeze(text_embedding.numpy())

def embed_image(
    query_image,
    data_processor,
    embedder
) -> list:
    
    # preprocess image
    processed_image = data_processor(
        images=query_image,
        text=None,
        return_tensors="tf"
    )["pixel_values"]

    # image embedding
    return np.squeeze(embedder.get_image_features(processed_image).numpy())

st.title("Text-to-Image Search App")

images_paths = glob.glob("data/samples/*.jpg")

# # make sure that dataset is downloaded
# if not images_paths:
#     with st.spinner("downloading dataset (around 4gb)..."):
#         download_extract_dataset()

#           images_paths = glob.glob("data/raw/*.jpg")
#         st.success("download and extract complete!")
# else:
#     st.success("found dataset!")

# get file names of images (dataset)

# define clip data processor
if "processor" not in st.session_state:
    st.session_state.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# define clip model
if "model" not in st.session_state:
    st.session_state.model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# initialize a variable to track if attachment changed
if "image_query" not in st.session_state:
    st.session_state.image_query = None

# initialize a variable to track if text changed
if "text_query" not in st.session_state:
    st.session_state.text_query = None

# initialize a variable to track if samples changed
if "num_of_samples" not in st.session_state:
    st.session_state.num_of_samples = None

# initialize a variable to save vector embeddings state
if "vector_db_client" not in st.session_state:
    st.session_state.vector_db_client = QdrantClient(
        host="localhost",
        port=6333
    )

# create collection for image embeddings
if "collection" not in st.session_state:
    st.session_state.collection = st.session_state.vector_db_client.recreate_collection(
    collection_name="images_embeddings",
    vectors_config=models.VectorParams(
        size=512, # size of embeddings
        distance=models.Distance.COSINE # similarity criteria
    )
)

num_of_samples = st.text_input(f"please enter a number of images to be embedded, maximum {len(images_paths)}")

if num_of_samples:

    # track if he query new text
    if num_of_samples != st.session_state.num_of_samples:

        # replace old with the new one 
        st.session_state.num_of_samples = num_of_samples

        with st.spinner(f"embedding {num_of_samples} images..."):
            # embed the dataset
            df = embed_dataset(
                num_of_samples=int(num_of_samples),
                images_paths=images_paths,
                data_processor=st.session_state.processor,
                embedder=st.session_state.model
            )

            # insert/update (if exist) embeddings to Qdrant
            st.session_state.vector_db_client.upsert(
                collection_name="images_embeddings",
                points=models.Batch(
                    ids=df["id"],
                    vectors=df["embedding"],
                    payloads=df["dir"]
                )
            )

            st.success("successfully saved embedding to Qdrant!")

    # input search query
    text_query = st.text_input("text search query:")

    # image search query
    image_query = st.file_uploader("image search query", type=["jpg", "jpeg", "png"])

    if text_query:

        # track if he query new text
        if text_query != st.session_state.text_query:

            # replace old with the new one 
            st.session_state.text_query = text_query
            st.session_state.image_query = None

            # embed input query text
            text_query_embedding = embed_text(
                text_query=text_query,
                data_processor=st.session_state.processor,
                embedder=st.session_state.model
            )

            # search for image
            results = st.session_state.vector_db_client.search(
                collection_name="images_embeddings",
                query_vector=text_query_embedding,
                limit=5
            )

            # show results
            st.subheader("top 5 images related to input text:")

            st.image(
                    image=[result.payload["dir"] for result in results], # images directories
                    caption=[f"Score: {result.score}" for result in results], # images scores
                    use_column_width=True,
                )

    if image_query:

        # track if he uploaded new image
        if image_query != st.session_state.image_query:

            # replace old with the new one 
            st.session_state.image_query = image_query
            st.session_state.text_query = None
            
            # show uploaded image
            st.image(
                image_query,
                caption="query image",
                use_column_width=True
            )

            # load image from disk
            image = Image.open(
                fp=image_query,
                mode="r"
            )

            # embed qeury image
            query_image_embedding = embed_image(
                query_image=image,
                data_processor=st.session_state.processor,
                embedder=st.session_state.model
            )

            # search for similar image
            results = st.session_state.vector_db_client.search(
                collection_name="images_embeddings",
                query_vector=query_image_embedding,
                limit=5
            )

            # show results
            st.subheader("top 5 images related to input image:")
            
            st.image(
                image=[result.payload["dir"] for result in results], # images directories
                caption=[f"Score: {result.score}" for result in results], # images scores
                use_column_width=True,
            )