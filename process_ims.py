from imageai.Detection import ObjectDetection
import os
import pandas as pd
import numpy as np
import glob
import math
import random
import shutil
from sklearn.metrics import classification_report
from PIL import Image



### load progress pic metadata from file

# df = pd.read_csv('download_all_progresspics_cumulative.csv')
df = pd.read_csv('download_all_progresspics_1219.csv')
chop = df['title'].str.split('[',expand=True)
sex_age_height = chop[0].str.split('/',expand=True)
df['is_female'] = sex_age_height[0]=='F'
df['age'] = sex_age_height[1]
feet_inches = sex_age_height[2].str.replace('‘', '\'').str.replace('”', '\'').str.replace('’', '\'').str.replace('"', '\'').str.split('\'',expand=True)
df['height'] = pd.to_numeric(feet_inches[0].str.extract('(\d+)', expand=False))*12+pd.to_numeric(feet_inches[1].str.extract('(\d+)', expand=False))

weight = chop[1].str.split(']',expand=True)[0]
weight_break = weight.str.split('&gt;',expand=True)
weight_1 = pd.to_numeric(weight_break[0].str.extract('(\d+)', expand=False))
weight_2 = pd.to_numeric(weight_break[1].str.split('=',expand=True)[0].str.extract('(\d+)', expand=False))
df['start_weight']=pd.concat([weight_1, weight_2]).max(level=0)
df['end_weight']=pd.concat([weight_1, weight_2]).min(level=0)


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


### load model for detecting persons in pictures
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


### crop out images of people and sort based on progress pic metadata

# data_directory = 'cleaned_images'
data_directory = 'holdout_set'
written = 0
for f in glob.glob('progresspics/*.jpeg'):
    data = df.loc['progresspics/'+df['file_name']==f,:]
    if len(data)!=1 or (data['start_weight'].isnull().any()) or (data['end_weight'].isnull().any()):
        continue
    sex_directory = 'male'
    if data['is_female'].values:
        sex_directory = 'female'
    weight_directory = str(int(np.floor(data['start_weight']/30)*30))
    next_weight_directory = str(int(np.floor(data['end_weight']/30)*30))
    
    w1 = str(int(data['start_weight'].values[0]))
    w2 = str(int(data['end_weight'].values[0]))

    try:
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , f))

        new_file_name = w1+'__'+w2
        reorder_detections = sorted([d for d in detections if d['name']=='person'],key=lambda x: x['box_points'][0])
        if(len(reorder_detections)==2):
            for i,d in enumerate(reorder_detections):
                if d['name']=='person':
                    directory = os.path.join(execution_path,data_directory,sex_directory,weight_directory)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    crop(os.path.join(execution_path , f),d['box_points'],os.path.join(directory,new_file_name+'_'+str(written)+'.jpeg'))
                    weight_directory = next_weight_directory
                    new_file_name = w2+'__'+w1
                    written = written + 1
    except:
        print('failed on '+f)
    print(str(written)+' written')



### move cleaned images into train/test file structure

min_val = 80
max_val = 400
n_per_group = 600
root_directory = 'cleaned_images/female/'
files = glob.glob(root_directory+'/*/*')
vals = [int(f.split('/')[-1].split('__')[0]) for f in files]

file_pattern = 'female_rerun/%s/%s/%s-%s-images/'
group = 0
done = 0
for v, f in sorted(zip(vals,files)):
    if v < max_val and v > min_val:
        train_test = 'train'
        if random.random()<0.2:
            train_test = 'test'
        new_loc = file_pattern % (train_test,group,group,train_test)
        if not os.path.exists(new_loc):
            os.makedirs(new_loc)
        shutil.copy2(f, new_loc)
        done = done + 1
        if done > n_per_group:
            done = 0
            group = group + 1




### train new model using progress pics

from imageai.Prediction.Custom import ModelTraining
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("female")

model_trainer.trainModel(num_objects=6, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)








##
#  testing on a holdout
##


### load custom trained model from file
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "male_model.h5"))
prediction.setJsonPath(os.path.join(execution_path, "male_model_class.json"))
prediction.loadModel(num_objects=4)

min_val = 100
max_val = 450
root_directory = 'holdout_set/male/'
files = glob.glob(root_directory+'/*/*')
vals = [int(f.split('/')[-1].split('__')[0]) for f in files]

bin_maxes = np.array([185,235,360,445]) # male 4
# bin_maxes = np.array([150,185,235,400]) # female 4

# bin_maxes = np.array([142,165,195,235,353,400])
results = []
for v, f in sorted(zip(vals,files)):
    if v < max_val and v > min_val:
        try:
            predictions, probabilities = prediction.predictImage(f, result_count=1)
            bn = str(np.where(v < bin_maxes)[0][0])
            results.append({'truth':bn, 'prediction':predictions[0], 'prob':probabilities[0]})
        except:
            print('uhoh {}'.format(f))
df_ev = pd.DataFrame.from_records(results)

print(classification_report(df_ev['truth'],df_ev['prediction']))
(df_ev['truth']==df_ev['prediction']).sum()/df_ev.shape[0]

np.sqrt((pd.to_numeric(df_ev['truth'])-pd.to_numeric(df_ev['prediction'])).pow(2).mean())
pd.to_numeric(df_ev['truth']).std()

# GETTING SOME PLACE DATA

# from googleplaces import GooglePlaces, types, lang

# YOUR_API_KEY = 'AIzaSyA4FeWD2cp4x_RZwbkO2K5jOw22bx1RfYQ'

# google_places = GooglePlaces(YOUR_API_KEY)

# query_result = google_places.nearby_search(
#         location='Chicago, IL', keyword='beach',
#         radius=20000)

# for i, place in enumerate(query_result.places):
#     print(place.name)
#     for j, photo in enumerate(place.photos):
#         photo.get(maxheight=500, maxwidth=500)
#         with open('tst'+str(i)+'_'+str(j)+'.jpg','wb') as f:
#             f.write(photo.data)

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

# https://wallethub.com/edu/fattest-cities-in-america/10532/
cities = ['memphis', 'akron', 'detroit', 'chicago', 'san francisco', 'denver']
for city in cities:
    response.download({'keywords':'"'+city+'" standing selfie','limit':100})


for city in cities:
    written = 0
    loc_directory = os.path.join(execution_path,'cleaned_images_city/'+city)

    for f in glob.glob('downloads/'+'"'+city+'" standing selfie'+'/*.jpg') + glob.glob('downloads/'+'"'+city+'" standing selfie'+'/*.jpeg'):
        try:
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , f))

            for i,d in enumerate(detections):
                if d['name']=='person' and d['percentage_probability']>50:
                    if not os.path.exists(loc_directory):
                        os.makedirs(loc_directory)
                    crop(os.path.join(execution_path , f),d['box_points'],os.path.join(loc_directory,str(written)+'.jpeg'))
                    written = written + 1
        except (ValueError, OSError):
            print('error on {}'.format(f))
        print(str(written)+' written')

from scipy import stats
results_city = {}
for city in cities:
    city_results = []
    for f in glob.glob('cleaned_images_city/'+city+'/*.jpeg'):
        predictions, probabilities = prediction.predictImage(f, result_count=1)
        city_results.append(int(predictions[0]))
    tst = np.array(city_results)
    results_city[city]={'mean':tst.mean(),'std':stats.sem(tst)}

results_city[cities]







#### picaday
picaday_directory = os.path.join(execution_path,'cleaned_images_picaday/')

for f in sorted(glob.glob('picaday/*.jpg')):
    try:
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , f))

        for i,d in enumerate(detections):
            if d['name']=='person' and d['percentage_probability']>50:
                if not os.path.exists(picaday_directory):
                    os.makedirs(picaday_directory)
                crop(os.path.join(execution_path , f),d['box_points'],os.path.join(picaday_directory,str(written)+'.jpeg'))
                written = written + 1
    except (ValueError, OSError):
        print('error on {}'.format(f))

