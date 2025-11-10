# Cross-modal Image Retrieval System

This project implements a smart image retrieval system using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. It enables users to search for images in a folder using natural language text queries, performing cross-modal retrieval to find visually similar images based on semantic meaning rather than just visual features.

## Features

- 🔍 **Text-to-Image Search**: Retrieve images based on natural language descriptions
- 🧠 **CLIP Model Integration**: Uses OpenAI's powerful vision-language model
- ⚙️ **Configurable Thresholds**: Adjustable confidence and similarity settings
- 🖼️ **Visual Results Display**: Shows matching images with confidence scores
- 📁 **Batch Processing**: Processes all images in a specified folder

## How It Works

1. **Model Loading**: Loads the CLIP ViT-B/32 model for encoding images and text
2. **Image Encoding**: Preprocesses and encodes all images in the specified folder
3. **Text Encoding**: Encodes the user's text query using the same CLIP model
4. **Similarity Calculation**: Computes cosine similarity between text and image embeddings
5. **Result Filtering**: Applies confidence and similarity gap thresholds
6. **Results Display**: Shows matching images with their confidence scores

## Demo

<video controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

You can see the system in action retrieving images based on text queries like "a cute furry pet playing in the park" or "red sports car on a mountain road".

## Requirements

- Python 3.7+
- PyTorch
- CLIP
- Streamlit
- PIL (Pillow)
- NumPy

## Installation

1. Install required packages:
   ```bash
   pip install torch clip pillow streamlit numpy
   ```

2. Install CLIP from OpenAI:
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## Usage

1. Create a folder with images you want to search through
2. Run the application:
   ```bash
   streamlit run cross_modal_retrieval.py
   ```
3. Enter the path to your image folder
4. Adjust confidence and similarity thresholds as needed
5. Enter a text query describing the image you're looking for
6. Click "Search" to retrieve matching images

## Parameters

- **Image Folder Path**: Directory containing images to search
- **Confidence Threshold**: Minimum similarity score for results (0.0-1.0)
- **Similarity Gap**: Maximum difference from top score (0.0-0.5)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Example Queries

- "a cute furry pet playing in the park"
- "red sports car on a mountain road"
- "person surfing on ocean waves at sunset"