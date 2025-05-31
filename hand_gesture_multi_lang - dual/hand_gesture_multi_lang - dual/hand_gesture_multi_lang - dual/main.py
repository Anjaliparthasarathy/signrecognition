import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from gtts import gTTS
import os

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

def tts(text, lang):
     file = gTTS(text = text, lang = lang)
     file.save("speak_func.mp3")
     playsound.playsound('speak_func.mp3', True)
     os.remove("speak_func.mp3")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera")
        break

    x, y, _ = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    
    # Language selection based on gesture
    if className in ['stop']:
        tamil_text = "நிறுத்து"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "ఆపండి"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['fist']:
        tamil_text = "எப்படி இருக்கிறீர்கள்"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "మీరు ఎలా ఉన్నారు"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['smile']:
        tamil_text = "புன்னகை"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "చిరునవ్వు"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['run']:
        tamil_text = "ஓடு"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "పరుగు"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['live long']:
        tamil_text = "நீண்ட காலம் வாழ்க"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "దీర్ఘకాలం జీవించండి"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['thumps up']:
        tamil_text = "வெற்றி"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "విజయం"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output1.mp3")
        os.system("start output1.mp3")
        print("done")
    elif className in ['call me']:
        tamil_text = "என்னை அழையுங்கள்"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "నాకు ఫోన్ చెయ్"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
    elif className in ['peace']:
        tamil_text = "சமாதானம்"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "శాంతి"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done") 
    elif className in ['thumps down']:
        tamil_text = "தோல்வி"
        language = 'ta' 
        tts = gTTS(text=tamil_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
        telugu_text = "వైఫల్యం"
        language = 'te' 
        tts = gTTS(text=telugu_text, lang=language, slow=False)
        tts.save("output.mp3")
        os.system("start output.mp3")
        print("done")
    
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
