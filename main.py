import importnb
import cv2 as cv
from Model_Utilities import *
from commonfunctions import *
import joblib

with importnb.Notebook():
    import character_segmentation
    import Segmentation
    

def segment_lines(image_path):
    output = Segmentation.segment(image_path)
    return output

def save_lines(lines, user_code):
    print(f"Saving lines for {user_code}")
    print("lines", lines)
    if not os.path.exists(f"Data/lines/{user_code}/"):
        os.makedirs(f"Data/lines/{user_code}/")
        
    for i, line in enumerate(lines):
        cv.imwrite(f"Data/lines/{user_code}/line_{i}.png", line[0])
    
    return f"Data/lines/{user_code}/"

def save_words(lines, user_code):
    if not os.path.exists(f"Data/words/{user_code}/"):
        os.makedirs(f"Data/words/{user_code}/")
        
    for i, line in enumerate(lines):
        if not os.path.exists(f"Data/words/{user_code}/line_{i}/"):
            os.makedirs(f"Data/words/{user_code}/line_{i}/")

        for j, word in enumerate(line[1]):
            cv.imwrite(f"Data/words/{user_code}/line_{i}/word_{j}.png", word)
            
    return f"Data/words/{user_code}/"

def save_characters(characters, user_code, line_num, word_num):
    if not os.path.exists(f"Data/characters/{user_code}/"):
        os.makedirs(f"Data/characters/{user_code}/")
        
    if not os.path.exists(f"Data/characters/{user_code}/line_{line_num}/"):
        os.makedirs(f"Data/characters/{user_code}/line_{line_num}/")
        
    if not os.path.exists(f"Data/characters/{user_code}/line_{line_num}/word_{word_num}/"):
        os.makedirs(f"Data/characters/{user_code}/line_{line_num}/word_{word_num}/")
        
    for i, character in enumerate(characters):
        #image = cv.imread(character, cv.IMREAD_GRAYSCALE)
        inverted = cv.bitwise_not(character)
        cv.imwrite(f"Data/characters/{user_code}/line_{line_num}/word_{word_num}/character_{i}.png", inverted)
        

def segment_characters(lines_path, user_code):
    line_files = os.listdir(lines_path)
    
    for line_file in line_files:
        line_path = f"{lines_path}/{line_file}"
        for word_file in os.listdir(line_path):
            word_path = f"{line_path}/{word_file}"
            #print(f"Segmenting characters for {word_path}")
            word_characters = character_segmentation.segment_characters(word_path)
            filtered_characters = character_segmentation.filter_characters(word_characters)
            save_characters(filtered_characters, user_code, int(line_file.split('_')[1]), int(word_file.split('_')[1].split('.')[0]))
            
                
def Run(path, classifier):
    numOfFeatures = 1052
    chars = ['ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي']
    charLabels =['ba', 'ta', 'tha', 'gim', 'ha', 'kha', 'dal' ,'thal', 'ra', 'zay', 'sin', 'shin', 'sad', 'dad', 'tah', 'za', 'ayn', 'gayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'non', 'haa', 'waw', 'ya']
    positionsLabels=['Beginning','End','Isolated','Middle']

    word = ''
    #classifier = joblib.load('classifier0.pkl')
    folder= getListOfFiles(path)
    count=0
    data1 = []
    for file in folder:
        img = cv.imread(file)
        img_resized = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
        gray_img = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        cropped = removeMargins(gray_img)
        binary_img = binary_otsus(gray_img, 0)
        features = getFeatures(binary_img)
        data1.append(features)
        data2 = np.array(data1).reshape(-1, numOfFeatures)
        prediction = classifier.predict(data2)
        count=count+1
        print(f"Prediction for letter {count}: {prediction[0]}")
        #cv.imshow('image',binary_img)
        char=labeltochar(prediction[0])
        word=word+char
        data1 = []

    print(word)
    print(' ')
    return word
    #increase dataset      
        
def get_prediction(user_code, model):
    if not os.path.exists(f"Data/characters/{user_code}/"):
        print("No characters found for user")
        return
    
    characters = []
    words = []
    lines = []
    for line in os.listdir(f"Data/characters/{user_code}/"):
        line_string = ""
        for word in os.listdir(f"Data/characters/{user_code}/{line}/"):
            word_string = Run(f"Data/characters/{user_code}/{line}/{word}", model)
            #reverse the word
            word_string = word_string[::-1]
            words.append(word_string)
            line_string += word_string + " "
        lines.append(line_string)
               
    return lines
            
        
        
def extract_lines_from_image(image, user_code, model):
    
    #save the image
    if not os.path.exists(f"Data/images/"):
        os.makedirs(f"Data/images/")
    
    image_path = f"Data/images/{user_code}.png"
    cv.imwrite(image_path, image)
    
    lines = segment_lines(image_path)
    save_lines(lines, user_code)
    words_path = save_words(lines, user_code)
    segment_characters(words_path, user_code)
    lines = get_prediction(user_code, model)

    return lines
    
# classifier = joblib.load('classifier0.pkl')  
# main('D:/UNI/CCE_sem_8_LAST_YAY_^^/gp2/project/Alfarahifi_org/arabic-ocr/paragraphs_per_user/paragraphs_per_user/user001/com_paragraph.png', "user_1", classifier)