#!/usr/bin/python3

from fake_useragent import UserAgent
import argparse
import colorama
import json
import os
import re
import requests
import sys
import numpy as np
import time
import pandas as pd


def get_valid_filename(s):
    ''' strips out special characters and replaces spaces with underscores, len 200 to avoid file_name_too_long error '''
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'[^\w.]', '', s)[:200]


def erase_previous_line():
    # cursor up one line
    sys.stdout.write("\033[F")
    # clear to the end of the line
    sys.stdout.write("\033[K")


def get_pictures_from_subreddit(data, output_file):
    subreddit = data['subreddit']
    image_url = data['image_url']
    if '.png' in image_url:
        extension = '.png'
    elif '.jpg' in image_url or '.jpeg' in image_url:
        extension = '.jpeg'
    elif 'imgur' in image_url:
        image_url += '.jpeg'
        extension = '.jpeg'
    else:
        return None

    # redirects = False prevents thumbnails denoting removed images from getting in
    image = requests.get(image_url, allow_redirects=False)
    if(image.status_code == 200):
        file_name = get_valid_filename(data['title']) + extension
        try:
            output_filehandle = open(
                subreddit + '/' + file_name, mode='bx')
            output_filehandle.write(image.content)
        except:
            pass
        return file_name
    return None


def main():
    parser = argparse.ArgumentParser(description='Fetch images from a subreddit (eg: python3 grab_pictures.py -s itookapicture CozyPlaces -n 100 -t all)')
    parser.add_argument('-s', '--subreddit', nargs='+', type=str, metavar='',
                        required=True, help='Exact name of the subreddits you want to grab pictures')
    parser.add_argument('-n', '--number', type=int, metavar='', default=50,
                        help='Optionally specify number of images to be downloaded (default=50)')
    parser.add_argument('-v', '--view', nargs='+', type=str, metavar='', choices=['top', 'new'],
                        default=['top','new'], help='Optionally pick whether you want [top, new] (default=[top,new])')
    parser.add_argument('-t', '--timeframe', nargs='+', type=str, metavar='', choices=['day', 'week', 'month', 'year', 'all'],
                        default=['day', 'week', 'month', 'year', 'all'], help='Optional timeframe [day, week, month, year or all] (default=day)')
    parser.add_argument('-p', '--previous_posts', type=str, help='Optional file containing titles of previous posts')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file for post info')
    args = parser.parse_args()

    subreddits = args.subreddit
    limit = int(args.number)
    timeframe = args.timeframe
    views = args.view
    previous_posts_file = args.previous_posts
    output_file = args.output_file

    previous_posts = []
    if previous_posts_file:
        previous_posts = pd.read_csv(previous_posts_file)['title'].values.tolist()


    colorama.init()
    ua = UserAgent()

    to_download = []
    for j in range(len(subreddits)):
        for k in range(len(views)):
            for t in range(len(timeframe)):
                after = ''
                count = 0
                while count < limit:
                    lim = min(limit-count,100)
                    count = count + lim
                    print('Getting '+str(lim)+' posts from r/' + subreddits[j])
                    url = 'https://www.reddit.com/r/' + subreddits[j] + '/' + views[k] + '/.json?sort='+ views[k] +'&t=' + \
                        timeframe[t] + '&limit=' + str(lim) + after
                    response = requests.get(url, headers={'User-agent': ua.random})

                    if not response.ok:
                        print("Error check the name of the subreddit", response.status_code)
                        exit()

                    if not os.path.exists(subreddits[j]):
                        os.mkdir(subreddits[j])
                    # notify connected and downloading pictures from subreddit
                    erase_previous_line()
                    print('downloading '+str(count-lim)+'-'+str(count)+' pictures from r/' + subreddits[j] + '..')
                    resp = response.json()
                    data = resp['data']['children']
                    for i in range(len(data)):
                        current_post = data[i]['data']
                        if current_post['title'] not in previous_posts:
                            to_download.append({
                                'subreddit':subreddits[j],
                                'image_url':current_post['url'],
                                'title':current_post['title'],
                                'score':current_post['score'],
                                'num_comments':current_post['num_comments'],
                                'author':current_post['author']})
                            previous_posts.append(current_post['title'])
                    if not resp['data']['after']:
                        break
                    after = '&after=' + resp['data']['after']
    for i in range(len(to_download)):
        erase_previous_line()
        print('.. ' + str((i*100)//len(to_download)) + '%')
        file_name = get_pictures_from_subreddit(to_download[i], output_file)
        to_download[i]['file_name']=file_name
    post_info = pd.DataFrame.from_records(to_download)
    post_info.loc[post_info['file_name'].notnull()].to_csv(output_file)
    print('Done')
    

if __name__ == '__main__':
    main()

# python3 grab_pictures.py -s progresspics -n 1000 -p download_all_progresspics_cumulative.csv -o download_all_progresspics_1203.csv
