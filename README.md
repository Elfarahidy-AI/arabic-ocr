## **Description**

This module is a core component of the ELFARAHIDI.AI project, designed to extract and recognize handwritten Arabic text from images. It utilizes a combination of image processing, text segmentation, feature extraction, and machine learning classification techniques to accurately convert handwritten Arabic into digital text.

## **Features**

**Image Preprocessing**: Cleans and prepares input images for segmentation.

**Text Segmentation**: Divides the text into lines, words, and individual characters. 
![image](https://github.com/Elfarahidy-AI/arabic-ocr/assets/65886084/a0e09727-203c-46f8-bb6e-138608c6f529)

**Feature Extraction**: Extracts relevant features from each character for classification.

**Classification**: Employs a Random Forest classifier to recognize characters based on extracted features.

## **Requirements**

Python 3.x

**Libraries**: OpenCV, NumPy, scikit-learn, etc. (See requirements.txt)

## **Installation**

**Clone the repository**: git clone [:link:](https://github.com/Elfarahidy-AI/arabic-ocr)

**Install dependencies**: pip install -r requirements.txt

## **Usage**

Place your handwritten Arabic text images in the input_images folder.

Run the main script: python main.py

The extracted text will be saved in the output_text folder.

## **Limitations**

Accuracy may vary depending on the quality of input images and handwriting style.

Currently, it does not support diacritics recognition.

## **Future Work**

Improve segmentation algorithms for better handling of connected and overlapping characters.

Incorporate deep learning models for potentially higher accuracy.

Add diacritics recognition functionality.
