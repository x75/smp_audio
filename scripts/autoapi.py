import urllib3
import json
import time, os
from pprint import pformat

from config import api_url, api_key
from config import smp_audioArgumentParser

headers = {'X-Authentication': api_key}

headers_json = {'Content-Type': 'application/json'}
headers_json.update(headers)

http = urllib3.PoolManager()

def ns2kw(ns):
    kw = dict([(_, getattr(ns, _)) for _ in dir(ns) if not _.startswith('_')])
    return kw

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
        if location.startswith('http'):
            call_url = location
        else:
            call_url = api_url + location
            
    else:
        call_url = api_url + '/files/' + location
        
    r = http.request(
        'GET',
        call_url,
        headers=headers,
    )
    if r.status == 200:
        open(os.path.basename(location), 'wb').write(r.data)
        print("downloaded {0}".format(location))

    res = {'status': r.status}
    if with_result:
        res.update(json.loads(r.data.decode('utf-8')))
    return res

def task_status(location):
    if location.startswith('http'):
        call_url = location
    else:
        call_url = api_url + location
        
    r = http.request(
        'GET',
        call_url,
        headers=headers,
    )
    print(r.data)
    res = json.loads(r.data.decode('utf-8'))
    return res

def autoedit_post(data):
    call_url = api_url + '/api/autoedit'
    encoded_data = json.dumps(data).encode('utf-8')
    print("call_url {0}".format(call_url))
    print("data {0}".format(data))
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers_json
    )
    print("r.status {0}".format(r.status))
    print("r.data {0}".format(r.data))
    res = json.loads(r.data.decode('utf-8'))
    return res

def autocover_post(data):
    call_url = api_url + '/api/autocover'
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
    call_url = api_url + '/api/automaster'
    encoded_data = json.dumps(data).encode('utf-8')
    print("call_url {0}".format(call_url))
    print("data {0}".format(data))
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers_json
    )
    print("r.status {0}".format(r.status))
    print("r.data {0}".format(r.data))
    res = json.loads(r.data.decode('utf-8'))
    return res

def main_api(args):
    """main_api

    Run autoedit workflow via API. Upload files, run the process,
    download output.
    """
    # convert args to dict to json
    data = ns2kw(args)
    # print("data = {0}".format(pformat(data)))

    ############################################################
    # upload the files
    print("uploading filenames {0}".format(data['filenames']))
    # would prefer this but hey
    # files = [('soundfile', (_, open(_, 'rb').read(), 'audio/wav')) for _ in data['filenames']]
    # print("files {0}".format(files))

    filenames_to_upload = [_ for _ in data['filenames']]
    if 'references' in data:
        filenames_to_upload += data['references']
        
    for filename in filenames_to_upload:
        files = [('soundfile', (os.path.basename(filename), open(filename, 'rb').read(), 'audio/wav'))]
        res = files_upload(files)
        print("res {0}".format(res))

    ############################################################
    # start the job
    print("starting job {0} with conf {1}".format(args.mode, data))
    if args.mode == 'autoedit':
        res = autoedit_post(data)
    elif args.mode == 'autocover':
        res = autocover_post(data)
    elif args.mode == 'automaster':
        res = automaster_post(data)
    # HACK
    if 'task' in res['data']:
        location = res['data']['task']['url']
    else:
        location = res['data']['url']
    print("autoapi     mode {0}".format(args.mode))
    print("autoapi response {0}".format(res))

    ############################################################
    # poll for output until 200 / not 404 or timeout
    print("waiting for output file to become ready for download on {0}".format(location))
    # download output file
    cnt = 0
    inprogress = True
    while inprogress and cnt < 100:
        print("req {0}".format(location))
        res = task_status(location)
        print("res {0}".format(pformat(res)))
        if res['data']['status'] == 'done':
            inprogress = False
        else:
            time.sleep(1)
        cnt += 1
        
    location = res['data']['url']
    print("autoapi downloading {0}".format(location))
    res = files_download(location, with_result=True, location_only=True)

    print("autoapi res {0}".format(res))

    if 'data' in res and 'output_files' in res['data']:
        for output_file in res['data']['output_files']:
            location = output_file['filename']
            print("autoapi download location {0}".format(location))
            res = files_download(location)
    
    return res

if __name__ == '__main__':
    parser = smp_audioArgumentParser()

    args = parser.parse_args()
    args.rootdir = './'

    main_api(args)
