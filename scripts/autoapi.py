import urllib3
import json
import time, os
from pprint import pformat

from smp_audio.cmd import smp_audioArgumentParser
from smp_audio.util import ns2kw

from config import api_url, api_key

headers = {'X-Authentication': api_key}

headers_json = {'Content-Type': 'application/json'}
headers_json.update(headers)

http = urllib3.PoolManager()

def files_upload(files):
    call_url = api_url + '/files'
    r = http.request(
        'POST',
        call_url,
        headers=headers,
        fields=files
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def files_download(location, with_result=False, location_only=False):
    if location_only:
        call_url = location
    else:
        call_url = api_url + '/files/' + location
        
    r = http.request(
        'GET',
        call_url,
        headers=headers,
    )
    if r.status == 200:
        open(os.path.basename(location), 'wb').write(r.data)
        print(f"downloaded {location}")

    res = {'status': r.status}
    if with_result:
        res.update(json.loads(r.data.decode('utf-8')))
    return res

def task_status(location):
    # call_url = api_url + '/files/' + location
    r = http.request(
        'GET',
        location,
        headers=headers,
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def autoedit_post(data):
    call_url = api_url + '/api/smp/autoedit'
    encoded_data = json.dumps(data).encode('utf-8')
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers_json
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def autocover_post(data):
    call_url = api_url + '/api/smp/autocover'
    encoded_data = json.dumps(data).encode('utf-8')
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers_json
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def automaster_post(data):
    call_url = api_url + '/api/smp/automaster'
    encoded_data = json.dumps(data).encode('utf-8')
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers_json
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def main_api(args):
    """main_api

    Run autoedit workflow via API. Upload files, run the process,
    download output.
    """
    # convert args to dict to json
    data = ns2kw(args)
    # print(f"data = {pformat(data)}")

    ############################################################
    # upload the files
    print(f"uploading filenames {data['filenames']}")
    # would prefer this but hey
    # files = [('soundfile', (_, open(_, 'rb').read(), 'audio/wav')) for _ in data['filenames']]
    # print(f'files {files}')

    filenames_to_upload = [_ for _ in data['filenames']]
    if 'references' in data:
        filenames_to_upload += data['references']
        
    for filename in filenames_to_upload:
        files = [('soundfile', (os.path.basename(filename), open(filename, 'rb').read(), 'audio/wav'))]
        res = files_upload(files)
        print(f'res {res}')

    ############################################################
    # start the job
    print(f'starting job {args.mode} with conf {data}')
    if args.mode == 'autoedit':
        res = autoedit_post(data)
    elif args.mode == 'autocover':
        res = autocover_post(data)
    elif args.mode == 'automaster':
        res = automaster_post(data)
    location = res['data']['task']['url']
    print(f'autoapi     mode {args.mode}')
    print(f'autoapi response {res}')

    ############################################################
    # poll for output until 200 / not 404 or timeout
    print(f"waiting for output file to become ready for download ...")
    # download output file
    cnt = 0
    inprogress = True
    while inprogress and cnt < 100:
        res = task_status(location)
        print(f"res {pformat(res)}")
        if res['data']['status'] == 'done':
            inprogress = False
        else:
            time.sleep(1)
        cnt += 1
        
    location = res['data']['url']
    print(f"autoapi downloading {location}")
    res = files_download(location, with_result=True, location_only=True)

    print(f"autoapi res {res}")

    if 'data' in res and 'output_files' in res['data']:
        for output_file in res['data']['output_files']:
            location = output_file['filename']
            print(f"autoapi download location {location}")
            res = files_download(location)
    
    return res

if __name__ == '__main__':
    parser = smp_audioArgumentParser()

    args = parser.parse_args()
    args.rootdir = './'

    main_api(args)
