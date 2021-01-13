import urllib3
import json
import time

from smp_audio.cmd import smp_audioArgumentParser
from smp_audio.util import ns2kw

api_url = 'http://127.0.0.1:5000'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {
    'User-Agent': user_agent,
    'X-Authentication': 'dt_bf04b14e8e86ad67c54fd4408c92a72181249db96c37939c2f8e9fc635782421',
}
headers_json = {
    'Content-Type': 'application/json'
}
headers_json.update(headers)

http = urllib3.PoolManager()

def files_upload(files):
    call_url = api_url + '/files'
    r = http.request(
        'POST',
        call_url,
        # body=encoded_data,
        headers=headers,
        fields=files
    )
    res = json.loads(r.data.decode('utf-8'))
    return res

def files_download(location):
    call_url = api_url + '/files/' + location
    # print(f'files_download call_url {call_url}')
    # print(f'files_download headers {headers}')
    r = http.request(
        'GET',
        call_url,
        headers=headers,
    )
    if r.status == 200:
        # res = json.loads(r.data.decode('utf-8'))
        open(location, 'wb').write(r.data)
        print(f"downloaded {location}")

    res = {'status': r.status}
    return res

def autoedit_post(data):
    call_url = api_url + '/api/smp/autoedit'
    # print(f'call url {call_url}')
    # print(f'autoedit_post headers {headers_json}')
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
    # print(f'call url {call_url}')
    # print(f'autoedit_post headers {headers_json}')
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
    # pop overlisting
    args.filenames = args.filenames[0]
    
    # convert args to dict to json
    data = ns2kw(args)

    ############################################################
    # upload the files
    print(f"uploading filenames {data['filenames']}")
    # would prefer this but hey
    # files = [('soundfile', (_, open(_, 'rb').read(), 'audio/wav')) for _ in data['filenames']]
    # print(f'files {files}')
    
    for filename in data['filenames']:
        files = [('soundfile', (filename, open(filename, 'rb').read(), 'audio/wav'))]
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
    location = res['data']['location']

    ############################################################
    # poll for output until 200 / not 404 or timeout
    print(f"waiting for output file to become ready for download ...")
    # download output file
    cnt = 0
    while True and cnt < 100:
        res = files_download(location)
        if res['status'] == 200:
            break
        else:
            time.sleep(1)
        cnt += 1
        
    return res

def main(args):
    # print(f'autoapi.main args {args}')
    if args.mode in ['autoedit']:
        res = main_api(args)
    elif args.mode in ['autocover']:
        res = main_autocover_api(args)
    elif args.mode in ['automaster']:
        res = main_automaster_api(args)

    print(f'autoapi.main res {res}')

if __name__ == '__main__':
    parser = smp_audioArgumentParser()

    args = parser.parse_args()
    args.rootdir = './'

    main_api(args)
